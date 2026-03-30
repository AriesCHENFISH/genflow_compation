import os
import re
import sys
import time
import csv
from datetime import datetime
from natsort import natsorted as na
import pandas as pd
from model import invoke_model_ds_rr


QUESTION_TYPES = ["global", "entity_action", "number_time", "relation", "conclusion"]


def simple_progress_bar(iterable, desc="", leave=True):
    """
    一个简单的基于 sys.stdout 的实时进度条
    """
    items = list(iterable)
    total = len(items)

    if total == 0:
        return

    start_time = time.time()

    for i, item in enumerate(items):
        yield item
        progress = (i + 1) / total
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        elapsed = time.time() - start_time

        sys.stdout.write(f"\r[*] {desc} |{bar}| {i + 1}/{total} [{elapsed:.1f}s]")
        sys.stdout.flush()

    if leave:
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()


class PipelineConfig:
    def __init__(self):
        # 1. 基础输入配置（按类别组织的文件夹）
        self.input_dir = "/home/users/chenxi84/chenxi/new_pipeline/categorized_files_real"

        # 类别映射（用于输出统计）
        self.categories = [
            "0_未分类或杂项",
            "1_长篇结构化应用文档",
            "2_学术论文与专业文献",
            "3_二维表格与日程数据",
            "4_极短视觉描述",
            "5_文学段落与扩写片段",
            "6_纯信息罗列清单",
        ]

        # 2. 实验配置：模型 x prompt 版本
        self.models_to_test = ["deepseek-v3"]
        self.compress_prompt_versions = ["anchor_v3"]

        # 3. 评测角色模型（建议固定，减少评测抖动）
        self.qa_generation_model = "deepseek-v3.2"
        self.qa_answering_model = "deepseek-v3.2"
        self.qa_scoring_model = "deepseek-v3.2"

        # 4. 压缩长度配置
        self.compress_n = 500
        self.compress_m = 1000

        # 5. 输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_base_dir = f"eval_output_{self.timestamp}"
        self.simplified_dir = os.path.join(self.output_base_dir, "1_simplified_docs")
        self.qa_gen_dir = os.path.join(self.output_base_dir, "2_qa_generated")
        self.qa_ans_dir = os.path.join(self.output_base_dir, "3_qa_answers")
        self.qa_score_dir = os.path.join(self.output_base_dir, "4_qa_scores")

        # 6. 统计文件
        models_str = "_".join(self.models_to_test)
        self.doc_csv_path = os.path.join(
            self.output_base_dir, f"{models_str}_scores_summary.csv"
        )
        self.exp_csv_path = os.path.join(
            self.output_base_dir, f"{models_str}_experiment_summary.csv"
        )
        self.time_csv_path = os.path.join(
            self.output_base_dir, f"{models_str}_simplify_time.csv"
        )

        # 7. QA生成目录（支持从已有目录读取）
        # 如果指定目录存在且不为空，则直接从该目录读取QA对
        self.qa_source_dir = "/home/users/chenxi84/chenxi/new_pipeline/eval_output_20260320_204329/2_qa_generated"

    def experiment_keys(self):
        for model in self.models_to_test:
            for prompt_version in self.compress_prompt_versions:
                yield model, prompt_version


class DocumentEvaluationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._create_dirs()
        # 按类别组织文件: {category: [files]}
        self.files_by_category = {}
        self.all_files = []  # 展平后的文件列表，每个元素为 (category, filename)
        if os.path.exists(self.config.input_dir):
            for category in self.config.categories:
                category_path = os.path.join(self.config.input_dir, category)
                if os.path.isdir(category_path):
                    files = [
                        f for f in na(os.listdir(category_path)) if f.endswith(".md")
                    ]
                    if files:
                        self.files_by_category[category] = files
                        for f in files:
                            self.all_files.append((category, f))

    def _create_dirs(self):
        for d in [
            self.config.output_base_dir,
            self.config.simplified_dir,
            self.config.qa_gen_dir,
            self.config.qa_ans_dir,
            self.config.qa_score_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            os.makedirs(
                os.path.join(self.config.simplified_dir, exp_name), exist_ok=True
            )
            os.makedirs(os.path.join(self.config.qa_ans_dir, exp_name), exist_ok=True)
            os.makedirs(os.path.join(self.config.qa_score_dir, exp_name), exist_ok=True)
            # 为每个实验创建类别子目录
            for category in self.config.categories:
                os.makedirs(
                    os.path.join(self.config.simplified_dir, exp_name, category),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(self.config.qa_ans_dir, exp_name, category),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(self.config.qa_score_dir, exp_name, category),
                    exist_ok=True,
                )

        if not os.path.exists(self.config.time_csv_path):
            with open(
                self.config.time_csv_path, "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["model", "prompt_version", "category", "file", "seconds"]
                )

    @staticmethod
    def _exp_name(model, prompt_version):
        return f"{model}__{prompt_version}"

    @staticmethod
    def _count_chars(text):
        return len(text) if isinstance(text, str) else 0

    def _extract_questions_only(self, content):
        """
        从 QA 文本中仅提取问题，并归一化成：
        Q1 [type=global]: 问题内容
        """
        question_lines = []
        pattern = re.compile(
            r"^\s*\*{0,2}Q(\d+)\s*\[type=([a-z_]+)\]\s*[:：]\*{0,2}\s*(.*)\s*$",
            re.IGNORECASE,
        )
        for line in content.splitlines():
            match = pattern.match(line.strip())
            if match:
                qid, qtype, qtext = match.groups()
                question_lines.append(f"Q{qid} [type={qtype.lower()}]: {qtext.strip()}")
        return "\n".join(question_lines)

    def _get_max_nums(self, content):
        n = self.config.compress_n
        m = self.config.compress_m
        if len(content) > n and len(content) <= m:
            return int(len(content) * 0.2)
        elif len(content) > m:
            return 500
        else:
            return None

    def _build_simplify_prompt(self, content, max_nums, prompt_version):
        if prompt_version == "baseline":
            return f"""# Role
你是一个极高素养的文档压缩与摘要专家。请将以下文本压缩至 **{max_nums} 字符左右**，在精简篇幅的同时，最大化保留信息完整性。

# 压缩要求
1. **首行全局概括**：首句必须概括全文性质、主题及组成部分，补全全局语境。
2. **结构化保留**：保留文档的逻辑骨架（如篇目编号、分段标识），确保不同模块边界清晰。
3. **关键要素补齐**：保留每一部分的"语境锚点"：身份（谁）、事件（因）、结果（果）。
4. **事实一致性**：核心数据、专业术语、专有名词必须精准，严禁张冠李戴。
5. **高效表达**：剔除抒情、客套和修饰词。使用客观短句，必要时利用符号提升信息密度。

# 字数控制策略
- 目标：压缩到 {max_nums} 字符以内，尽量接近，但不得超出。
- 当字数富余时：补充核心事件的具体描述或关键细节。
- 当字数超限时：按"修饰语 > 衔接词 > 辅助性语句"的顺序裁撤，但必须保留全局概括和结构标识。

# 待压缩文本
{content}

# 输出结果
请直接输出压缩后的文本，不允许输出任何其他信息和说明。
"""

        if prompt_version == "anchor_v3":
            return f"""将以下文本压缩至 **{max_nums} 字符以内**，直接输出结果，禁止任何解释或前缀。

# 压缩规则（严格执行）
- **保留结构**：原文分篇则分篇输出，保留标题、编号、页码等边界；禁止融合多篇。
- **保留关键数据**：专有名词、数字、结果必须原样保留。
- **精简表达**：删除修饰词、客套话、举例；用短句，但禁止大量斜杠拼凑。
- **表格处理**：表格转成紧凑的逗号分隔文本，保留行/列归属。

# 底线
- 字符数 ≤ {max_nums}
- 核心事实不变，不重构原文类别。

# 文本
{content}"""
            
            
        if prompt_version == "anchor_v2":
            return f"""你是一个极高素养且【对输出延迟极度敏感】的上下文压缩引擎。请将以下文本压缩至 **{max_nums} 字符以内**，最大化保留信息完整性、逻辑边界和下游生成潜能。

# ⚡ 最高执行指令：反思维链与零废话
必须直接、立刻输出最终的压缩结果！绝对禁止任何形式的内部推理、分析步骤；禁止输出"好的"、"分析如下"等任何前缀后缀。首个字符必须是压缩正文！

# 自适应压缩策略（隐式判断文档真实结构，对号入座，切忌生硬融合）
1. 【学术论文与专业文献】：必须保留完整的逻辑骨架（如：研究目的→核心方法→关键参数→结论）。允许使用简短的小标题或基础序号来保持结构边界清晰，**严禁**将其揉捏成不分段的密集文字块。
2. 【多篇集合与并列项】（如多份总结/检讨的汇总）：**必须保持各篇目的独立性**。采用极简列表形式（如"篇1：[核心]；篇2：[核心]"）。**绝对禁止**将多个独立主体的事件糅合成一个统一的叙述或宏观报告。
3. 【高密度数据与映射关系】（如带页码的生字表、课表）：必须保留原有的层级/分组映射关系（如"第X页："或"星期X："）。组内可使用紧凑逗号分隔，但严禁将不同分组的数据全部拍平混淆。
4. 【结构化模板与大纲】（如工作总结模板、课程大纲）：保留原有的模块维度划分。禁止为了压缩而使用大量斜杠（/）生硬拼接词语，必须保持基本的语意连贯性。
5. 【一般连续性文本】（普通故事、文章）：采用"语义折叠"。剔除客套话、修饰词和发散举例。使用客观短句，紧凑输出。

# 字数与底线约束
- 目标限制：绝对不能超出 {max_nums} 字符。在保真前提下越短越好。
- 事实保真：核心数据、专有名词必须精准。压缩是提炼而非重构，绝不能改变原文的客观存在形式（如把文献摘要变成故事，或把合集变成单篇）。

# 待压缩文本
{content}"""
            
            
            
            
            
            
            
            
            
#             """你是一个极高素养且【对输出延迟极度敏感】的上下文压缩引擎。请将以下文本压缩至 **{max_nums} 字符以内**，最大化保留信息完整性和下游生成潜能。

# # ⚡ 最高执行指令：反思维链与零废话（为了极致性能）
# 你必须直接、立刻输出最终的压缩结果。绝对禁止任何形式的内部推理、分析步骤或分类说明；禁止输出诸如"好的"、"根据要求"、"本文属于XX类型"等任何前缀、后缀。你的首个输出字符必须是压缩文本的实质内容！

# # 自适应压缩策略（隐式判断，直接执行）
# 1. 【连续性文本】（如论文、故事、文章）：采用"极限语义折叠"。剔除一切客套话、修饰词、发散性举例。只保留核心论点、因果、时间线和关键参数。合并段落，使用极其紧凑的客观短句，**严禁**强行输出带有多级缩进的 Markdown 列表（极其浪费 Token）。
# 2. 【高密度数据】（如生字表、课表、代码）：严禁概括。采用"极简降维"：直接输出紧凑的逗号分隔符（CSV）或极简字典。**绝对禁止使用 Markdown 表格（如 `|---|---|`）**，因为表格的边界符号会极大地增加输出 Token 和耗时。去除所有多余换行和空格，确保具体数据点（生字/时间/数值）零丢失。

# # 字数与底线约束
# - 目标：绝对不能超出 {max_nums} 字符。在不丢失核心数据的前提下，**字数越少越好，越短评价越高**。
# - 事实一致：核心数据、专业术语必须精准，严禁幻觉。
# - 绝对不扩写：如果原文短于 {max_nums}，压缩后的字数必须显著少于原文。

# # 待压缩文本
# {content}"""

        raise ValueError(f"未知的 prompt_version: {prompt_version}")

    def _simplify_md(self, content, model_type, prompt_version):
        max_nums = self._get_max_nums(content)
        if max_nums is None:
            return None
        prompt = self._build_simplify_prompt(content, max_nums, prompt_version)
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _generate_qa_md(self, content, model_type):
        prompt = f"""# Role
你是一个资深的阅读理解评测专家与信息提取大师。你的任务是基于【原始文本】生成一组固定题型、固定数量、可客观核验的 QA 对，用于评估压缩文本是否保留了用于下游复杂任务（如改写、分析、总结、检索）的核心价值。

# 任务目标
你的问题必须优先命中原文中最具价值、最容易在压缩中丢失或因过度概括而破坏的核心信息。如果是高密度表格/生字表，必须包含对具体条目的精准点查。

# 题型设计（必须严格遵守 8 个 QA 对）
- Q1-Q2: type=global（宏观把握：全文的核心主旨、最终结论、或者这篇材料的整体教学/表达目标是什么？）
- Q3-Q4: type=entity_action（关键细节：核心主体在特定情境下的特定动作、约束条件，或专有名词的具体指向。）
- Q5-Q6: type=number_time（高密检索/数据：最关键的数字、时间节点。若原文是生字表/课表等，请直接提问某个字/某个时间点的具体细节，如“鹤字的部首是什么？”）
- Q7: type=relation（逻辑与因果：事物之间的推演关系、转折原因、或者前提条件是什么？）
- Q8: type=conclusion（生成性延伸/产出：基于该文档的最终产出、核心价值，或如果用它做参考，能得出什么核心判断？）

# 出题要求
1. 问题必须覆盖全文核心价值，拒绝边缘无用的细枝末节。
2. 答案必须客观、唯一、可直接在原文中找到。
3. 对于结构化数据（如计划表、生字册），必须直接考察里面的具体数据点，严禁泛泛而谈。
4. 严格按照下方格式输出，不要输出任何解释，不要省略或增加题数。

# 输出格式（必须严格遵守，请勿使用 Markdown 代码块包裹，直接输出纯文本）
Q1 [type=global]: [问题]
A1: [答案]

Q2 [type=global]: [问题]
A2: [答案]

Q3 [type=entity_action]: [问题]
A3: [答案]

Q4 [type=entity_action]: [问题]
A4: [答案]

Q5 [type=number_time]: [问题]
A5: [答案]

Q6 [type=number_time]: [问题]
A6: [答案]

Q7 [type=relation]: [问题]
A7: [答案]

Q8 [type=conclusion]: [问题]
A8: [答案]

# 原始文本
{content}"""
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _answer_qa_md(self, content, q_content, model_type):
        time.sleep(1)
        prompt = f"""
Role: 你是一个高效的信息检索员。
Task: 请仅根据提供的【压缩文本】回答问题。

输入：
【压缩文本】：{content}

【待回答问题】：
{q_content}

回答准则：
1. 所有答案必须直接来源于【压缩文本】。
2. 不允许使用外部知识，不允许猜测。
3. 如果压缩文本中没有足够信息，请直接回答：信息缺失。
4. 回答要尽量简短，保留必要事实。

输出格式（必须严格遵守）：
Q1 [type=global]: [原问题]
A1: [你的回答]
"""
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _score_qa_md(self, reference_qa, answer_qa, model_type):
        time.sleep(1)
        prompt = f"""请对比【参考答案】与【实际回答】，逐题评估压缩文本是否保留了原文核心信息，给出 0-2 分及错误类型。

【参考答案】：
{reference_qa}

【实际回答】：
{answer_qa}

# 评分与错误类别
- 2 (correct): 核心事实完全匹配，信息完整。
- 1 (partial): 部分正确，缺限定词、细节，或轻微错配。
- 0 (omission/wrong): 回答错误、矛盾，或直接回答未提及/缺失（若为未提及缺失，错误类型必须标为 omission）。

# ⚡ 输出约束（最高优先级）
不要输出任何分析、理由或计算过程！不要计算总分！请直接、极其严格地按以下纯文本格式输出 8 行结果（必须保留题号和 type 标签）：

Q1 [type=global]: 2 | correct
Q2 [type=global]: 1 | partial
Q3 [type=entity_action]: 0 | omission
...
Q8 [type=conclusion]: 2 | correct
"""

        # """
        # Role: 你是一个极其严苛的文本对齐审计专家。
        # Task: 对比【参考答案】与【实际回答】，逐题判断压缩文本是否保留了原文核心信息，并给出分数与错误类型。

        # 输入：
        # 【参考答案（基于原文）】：
        # {reference_qa}

        # 【实际回答（基于压缩文）】：
        # {answer_qa}

        # 评分标准：
        # - 2分：回答与参考答案在核心事实层面完全匹配，关键信息完整。
        # - 1分：回答部分正确，但存在信息缺失、限定词缺失、数字不全、要素不完整等问题。
        # - 0分：回答错误、张冠李戴、与原文矛盾、臆造，或直接回答"信息缺失"。

        # 错误类型（每题只能选一个最主要的）：
        # - correct：完全正确
        # - omission：信息缺失 / 压缩文未提供答案
        # - partial：部分正确但信息不完整
        # - contradiction：与参考答案冲突
        # - entity_confusion：主体/对象/关系错配
        # - number_error：数字、时间、比例错误或不完整
        # - hallucination：编造原文没有的信息
        # - other：其他错误

        # 判定要求：
        # 1. 必须逐题打分。
        # 2. 如果回答是"信息缺失"，error_type 必须标为 omission，得分必须为 0。
        # 3. 如果只是缺少部分限定词或细节，通常应为 1 分，error_type 标为 partial 或 number_error。
        # 4. 必须保留题号和 type 标签。
        # 5. 最后给出总分 all 和 omission_count。

        # 输出格式（必须严格遵守）：
        # Q1 [type=global]: 2 | correct | [简短理由]
        # Q2 [type=global]: 1 | partial | [简短理由]
        # Q3 [type=entity_action]: 0 | omission | [简短理由]
        # ...
        # Q8 [type=conclusion]: 2 | correct | [简短理由]

        # all: X
        # omission_count: Y
        # """
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    @staticmethod
    def _parse_score_content(score_text):
        q_pattern = re.compile(
            r"^Q(\d+)\s*\[type=([a-z_]+)\]\s*:\s*([012])\s*\|\s*([a-z_]+)",
            re.IGNORECASE,
        )
        all_pattern = re.compile(r"^all\s*:\s*(\d+)\s*$", re.IGNORECASE)
        omission_pattern = re.compile(r"^omission_count\s*:\s*(\d+)\s*$", re.IGNORECASE)

        question_count = 0
        parsed_total = None
        omission_count = 0
        type_score_map = {q_type: 0 for q_type in QUESTION_TYPES}
        type_count_map = {q_type: 0 for q_type in QUESTION_TYPES}

        for raw_line in score_text.splitlines():
            line = raw_line.strip()
            q_match = q_pattern.match(line)
            if q_match:
                _, qtype, score, error_type = q_match.groups()
                qtype = qtype.lower()
                score = int(score)
                question_count += 1
                if qtype in type_score_map:
                    type_score_map[qtype] += score
                    type_count_map[qtype] += 1
                if error_type.lower() == "omission":
                    omission_count += 1
                continue

            total_match = all_pattern.match(line)
            if total_match:
                parsed_total = int(total_match.group(1))
                continue

            omission_match = omission_pattern.match(line)
            if omission_match:
                omission_count = int(omission_match.group(1))

        computed_total = sum(type_score_map.values())
        all_score = parsed_total if parsed_total is not None else computed_total
        max_score = question_count * 2
        return {
            "all_score": all_score,
            "max_score": max_score,
            "question_count": question_count,
            "omission_count": omission_count,
            "type_score_map": type_score_map,
            "type_count_map": type_count_map,
        }

    def run_step_1_simplify(self):
        print(
            f"\n[Step 1/5] 开始对原始文档进行压缩 (共 {len(self.all_files)} 篇文档)..."
        )
        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            print(f"  -> 测试实验: {exp_name}")
            for category, file in simple_progress_bar(
                self.all_files, desc="    压缩进度", leave=True
            ):
                raw_path = os.path.join(self.config.input_dir, category, file)
                save_path = os.path.join(
                    self.config.simplified_dir, exp_name, category, file
                )

                if os.path.exists(save_path):
                    continue

                with open(raw_path, "r", encoding="utf-8") as f:
                    content_raw = f.read()

                start_time = time.time()
                content_simplified = self._simplify_md(
                    content_raw, model, prompt_version
                )

                if content_simplified is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content_simplified)

                    with open(
                        self.config.time_csv_path, "a", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                model,
                                prompt_version,
                                category,
                                file,
                                round(time.time() - start_time, 3),
                            ]
                        )
        print("  -> Step 1 完成。")

    def _check_existing_qa_files(self):
        """
        检查qa_source_dir目录是否存在且包含QA文件
        返回: True 如果存在可用文件, False 否则
        """
        if not os.path.exists(self.config.qa_source_dir):
            return False

        # 检查目录中是否有md文件
        existing_files = [
            f for f in os.listdir(self.config.qa_source_dir) if f.endswith(".md")
        ]
        return len(existing_files) > 0

    def _copy_qa_from_source(self, file):
        """
        从源目录复制QA文件到当前输出目录
        """
        source_path = os.path.join(self.config.qa_source_dir, file)
        dest_path = os.path.join(self.config.qa_gen_dir, file)

        if os.path.exists(source_path) and not os.path.exists(dest_path):
            with open(source_path, "r", encoding="utf-8") as f:
                content = f.read()
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False

    def run_step_2_generate_qa(self):
        print(f"\n[Step 2/5] 基于原始文档生成固定题型 QA 对...")

        # 检查是否存在已生成的QA文件目录
        use_existing_qa = self._check_existing_qa_files()

        if use_existing_qa:
            print(f"  -> 检测到已有QA生成目录: {self.config.qa_source_dir}")
            print(f"  -> 模式: 从已有目录复制QA对")
            copied_count = 0
            for category, file in simple_progress_bar(
                self.all_files, desc="    复制进度", leave=True
            ):
                save_path = os.path.join(self.config.qa_gen_dir, file)

                if os.path.exists(save_path):
                    continue

                if self._copy_qa_from_source(file):
                    copied_count += 1
            print(f"  -> 成功复制 {copied_count} 个QA文件")
        else:
            print(f"  -> 未检测到已有QA生成目录，将重新生成QA对")
            model = self.config.qa_generation_model
            print(f"  -> QA 生成模型: {model}")
            for category, file in simple_progress_bar(
                self.all_files, desc="    生成进度", leave=True
            ):
                raw_path = os.path.join(self.config.input_dir, category, file)
                save_path = os.path.join(self.config.qa_gen_dir, file)

                if os.path.exists(save_path):
                    continue

                with open(raw_path, "r", encoding="utf-8") as f:
                    content_raw = f.read()

                qa_content = self._generate_qa_md(content_raw, model)
                if qa_content is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(qa_content)
        print("  -> Step 2 完成。")

    def run_step_3_answer_qa(self):
        print(f"\n[Step 3/5] 使用压缩文档作答 QA 对...")
        model_a = self.config.qa_answering_model
        print(f"  -> QA 答题模型: {model_a}")
        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            print(f"  -> 处理 {exp_name} 的压缩结果")
            for category, file in simple_progress_bar(
                self.all_files, desc="    答题进度", leave=True
            ):
                simplified_path = os.path.join(
                    self.config.simplified_dir, exp_name, category, file
                )
                qa_gen_path = os.path.join(self.config.qa_gen_dir, file)
                save_path = os.path.join(
                    self.config.qa_ans_dir, exp_name, category, file
                )

                if not os.path.exists(simplified_path) or not os.path.exists(
                    qa_gen_path
                ):
                    continue
                if os.path.exists(save_path):
                    continue

                with open(simplified_path, "r", encoding="utf-8") as f:
                    content_simplified = f.read()
                with open(qa_gen_path, "r", encoding="utf-8") as f:
                    q_content_raw = f.read()
                    q_content = self._extract_questions_only(q_content_raw)

                if not q_content:
                    continue

                answer_content = self._answer_qa_md(
                    content_simplified, q_content, model_a
                )
                if answer_content is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(answer_content)
        print("  -> Step 3 完成。")

    def run_step_4_score_qa(self):
        print(f"\n[Step 4/5] 评估回答准确度并打分...")
        model_s = self.config.qa_scoring_model
        print(f"  -> 裁判打分模型: {model_s}")
        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            print(f"  -> 评估 {exp_name} 的答卷")
            for category, file in simple_progress_bar(
                self.all_files, desc="    打分进度", leave=True
            ):
                qa_ans_path = os.path.join(
                    self.config.qa_ans_dir, exp_name, category, file
                )
                qa_gen_path = os.path.join(self.config.qa_gen_dir, file)
                save_path = os.path.join(
                    self.config.qa_score_dir, exp_name, category, file
                )

                if not os.path.exists(qa_ans_path) or not os.path.exists(qa_gen_path):
                    continue
                if os.path.exists(save_path):
                    continue

                with open(qa_ans_path, "r", encoding="utf-8") as f:
                    content_a = f.read()
                with open(qa_gen_path, "r", encoding="utf-8") as f:
                    content_ref = f.read()

                score_content = self._score_qa_md(content_ref, content_a, model_s)
                if score_content is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(score_content)
        print("  -> Step 4 完成。")

    def run_step_5_aggregate_scores(self):
        print(f"\n[Step 5/5] 汇总分数并生成统计报表...")
        doc_rows = []

        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            score_dir = os.path.join(self.config.qa_score_dir, exp_name)
            if not os.path.exists(score_dir):
                continue

            record_count = 0
            for category, file in self.all_files:
                score_file = os.path.join(score_dir, category, file)

                if not os.path.exists(score_file):
                    continue

                with open(score_file, "r", encoding="utf-8") as f:
                    score_text = f.read()

                parsed = self._parse_score_content(score_text)
                all_score = parsed["all_score"]
                max_score = parsed["max_score"]
                question_count = parsed["question_count"]
                omission_count = parsed["omission_count"]

                row = {
                    "file": file,
                    "category": category,
                    "model": model,
                    "prompt_version": prompt_version,
                    "score_rate": round(all_score / max_score, 6)
                    if max_score > 0
                    else None,
                    "omission_rate": round(omission_count / question_count, 6)
                    if question_count > 0
                    else None,
                }

                doc_rows.append(row)
                record_count += 1

            print(f"  -> 实验 {exp_name}: 成功统计 {record_count} 份记录。")

        if not doc_rows:
            print("  -> 未发现可汇总记录。")
            return

        df_doc = pd.DataFrame(doc_rows)
        df_doc.to_csv(self.config.doc_csv_path, index=False)

        # 实验级汇总（只保留要求的三项核心指标）
        df_exp = (
            df_doc.groupby(["model", "prompt_version"], as_index=False)
            .agg({"score_rate": "mean", "omission_rate": "mean"})
            .rename(
                columns={
                    "model": "模型",
                    "prompt_version": "Prompt版本",
                    "score_rate": "总体平均得分率",
                    "omission_rate": "信息缺失率",
                }
            )
        )

        # 计算各类文档平均得分率
        df_cat = df_doc.groupby(
            ["model", "prompt_version", "category"], as_index=False
        ).agg({"score_rate": "mean"})

        for category in self.config.categories:
            cat_data = df_cat[df_cat["category"] == category]
            if len(cat_data) > 0:
                for _, row in cat_data.iterrows():
                    mask = (df_exp["模型"] == row["model"]) & (
                        df_exp["Prompt版本"] == row["prompt_version"]
                    )
                    if mask.any():
                        df_exp.loc[mask, f"{category}_平均得分率"] = row["score_rate"]

        df_exp.to_csv(self.config.exp_csv_path, index=False)

        print(f"  -> 文档级统计 CSV 已输出至: {self.config.doc_csv_path}")
        print(f"  -> 实验级统计 CSV 已输出至: {self.config.exp_csv_path}")
        print("  -> Step 5 完成。")

    def run_all(self):
        print("===========================================================")
        print("  LLM 文本无损压缩评估流水线")
        print(f"  启动时间: {self.config.timestamp}")
        print(f"  输入目录: {self.config.input_dir}")
        print(f"  输出目录: {self.config.output_base_dir}")
        print(f"  压缩模型: {', '.join(self.config.models_to_test)}")
        print(f"  Prompt 版本: {', '.join(self.config.compress_prompt_versions)}")
        print("===========================================================\n")

        if not self.all_files:
            print(
                f"[警告] 未在 {self.config.input_dir} 发现任何 .md 文件，流水线终止。"
            )
            return

        self.run_step_1_simplify()
        self.run_step_2_generate_qa()
        self.run_step_3_answer_qa()
        self.run_step_4_score_qa()
        self.run_step_5_aggregate_scores()

        print("\n===========================================================")
        print("  🎉 流水线执行完毕！")
        print(f"  📂 所有数据保存在: {self.config.output_base_dir}")
        print("===========================================================\n")


if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = DocumentEvaluationPipeline(config)
    pipeline.run_all()
