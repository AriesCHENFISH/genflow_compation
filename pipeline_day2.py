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
        # 1. 基础输入配置
        self.input_dir = "all_files"

        # 2. 实验配置：模型 x prompt 版本
        self.models_to_test = ["qwen3.5-27b"]
        self.compress_prompt_versions = ["baseline", "anchor_v1"]

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
        self.doc_csv_path = os.path.join(self.output_base_dir, "scores_summary.csv")
        self.exp_csv_path = os.path.join(self.output_base_dir, "experiment_summary.csv")
        self.time_csv_path = os.path.join(self.output_base_dir, "simplify_time.csv")

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
        self.files = []
        if os.path.exists(self.config.input_dir):
            self.files = [
                f for f in na(os.listdir(self.config.input_dir)) if f.endswith(".md")
            ]

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
            os.makedirs(os.path.join(self.config.simplified_dir, exp_name), exist_ok=True)
            os.makedirs(os.path.join(self.config.qa_ans_dir, exp_name), exist_ok=True)
            os.makedirs(os.path.join(self.config.qa_score_dir, exp_name), exist_ok=True)

        if not os.path.exists(self.config.time_csv_path):
            with open(self.config.time_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["model", "prompt_version", "file", "seconds"])

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

        if prompt_version == "anchor_v1":
            return f"""# Role
你是一个零损信息压缩器。你的目标不是写摘要，而是在 **{max_nums} 字符以内** 最大化保留后续问答可验证的事实、结构与关系。

# 核心原则
1. **唯一事实来源**：只能依据待压缩文本，禁止引入外部知识。
2. **锚点优先保留**：优先保留文档类型/主题、结构边界、主体、动作、对象、时间、数字、比例、专有名词、结论、约束条件。
3. **禁止错配**：严禁把 A 的动作、数字、结论挂到 B 身上。
4. **缺失不补**：原文没有的信息，宁可省略，也不要脑补。
5. **表达压缩，不压缩事实**：可以压缩措辞，但不得压缩关键事实。

# 内部工作流（内部执行，不要输出过程）
1. 先识别全文必须保留的信息锚点：
   - 文档是什么、讨论什么、由几部分组成
   - 每个部分的主体｜动作/事件｜对象｜结果
   - 时间、数字、比例、专有名词
   - 因果、条件、约束、最终结论
2. 再进行压缩表达。
3. 输出前自检：
   - 是否遗漏关键数字、时间、专名
   - 是否混淆不同模块主体
   - 是否写入原文没有的信息

# 输出格式要求
1. 首行必须给出全文定位：文档性质 + 主题 + 组成部分。
2. 后续尽量按模块输出，每个模块优先采用：
   **[模块名/序号] 主体｜事件/动作｜关键条件/数据｜结果**
3. 允许使用"：" "；" "｜" "→" 提升密度。
4. 删除修饰语、客套语、重复解释，但保留事实链。
5. 总长度不得超过 {max_nums} 字符。

# 待压缩文本
{content}

# 输出结果
请直接输出压缩后的文本，不要输出任何解释、标题或附注。
"""

        raise ValueError(f"未知的 prompt_version: {prompt_version}")

    def _simplify_md(self, content, model_type, prompt_version):
        max_nums = self._get_max_nums(content)
        if max_nums is None:
            return None
        prompt = self._build_simplify_prompt(content, max_nums, prompt_version)
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _generate_qa_md(self, content, model_type):
        prompt = f"""
# Role
你是一个资深的阅读理解评测专家与信息提取大师。你的任务是基于【原始文本】生成一组固定题型、固定数量、可客观核验的 QA 对，用于评估压缩文本的信息保留质量。

# 任务目标
后续会使用压缩文本回答这些问题，因此你的问题必须优先命中原文中最重要、最容易在压缩中丢失或错配的信息。

# 题型设计（必须严格遵守）
请固定生成 **8 个 QA 对**，按以下分布：
- Q1-Q2: type=global（全文性质、主题、结构、组成部分）
- Q3-Q4: type=entity_action（谁做了什么、对象是什么、发生了什么）
- Q5-Q6: type=number_time（时间、数字、比例、数量、金额、年份等）
- Q7: type=relation（因果、条件、约束、前后关系）
- Q8: type=conclusion（最终结论、结果、判断、产出）

# 出题要求
1. 问题必须覆盖全文核心信息，而不是边缘细节。
2. 问题必须客观、唯一、可直接在原文中找到答案。
3. number_time 题必须优先问最关键的数字或时间，不要问无关数字。
4. relation 题必须问清楚"为什么 / 在什么条件下 / 由于什么导致什么"。
5. conclusion 题必须问最终结果或结论，不能与 global 重复。
6. 不要输出任何解释，不要省略任何题。

# 输出格式（必须严格遵守）
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
{content}
"""
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
        prompt = f"""
Role: 你是一个极其严苛的文本对齐审计专家。
Task: 对比【参考答案】与【实际回答】，逐题判断压缩文本是否保留了原文核心信息，并给出分数与错误类型。

输入：
【参考答案（基于原文）】：
{reference_qa}

【实际回答（基于压缩文）】：
{answer_qa}

评分标准：
- 2分：回答与参考答案在核心事实层面完全匹配，关键信息完整。
- 1分：回答部分正确，但存在信息缺失、限定词缺失、数字不全、要素不完整等问题。
- 0分：回答错误、张冠李戴、与原文矛盾、臆造，或直接回答"信息缺失"。

错误类型（每题只能选一个最主要的）：
- correct：完全正确
- omission：信息缺失 / 压缩文未提供答案
- partial：部分正确但信息不完整
- contradiction：与参考答案冲突
- entity_confusion：主体/对象/关系错配
- number_error：数字、时间、比例错误或不完整
- hallucination：编造原文没有的信息
- other：其他错误

判定要求：
1. 必须逐题打分。
2. 如果回答是"信息缺失"，error_type 必须标为 omission，得分必须为 0。
3. 如果只是缺少部分限定词或细节，通常应为 1 分，error_type 标为 partial 或 number_error。
4. 必须保留题号和 type 标签。
5. 最后给出总分 all 和 omission_count。

输出格式（必须严格遵守）：
Q1 [type=global]: 2 | correct | [简短理由]
Q2 [type=global]: 1 | partial | [简短理由]
Q3 [type=entity_action]: 0 | omission | [简短理由]
...
Q8 [type=conclusion]: 2 | correct | [简短理由]

all: X
omission_count: Y
"""
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    @staticmethod
    def _parse_score_content(score_text):
        q_pattern = re.compile(
            r"^Q(\d+)\s*\[type=([a-z_]+)\]\s*:\s*([012])\s*\|\s*([a-z_]+)\s*\|",
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
        print(f"\n[Step 1/5] 开始对原始文档进行压缩 (共 {len(self.files)} 篇文档)...")
        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            print(f"  -> 测试实验: {exp_name}")
            for file in simple_progress_bar(self.files, desc="    压缩进度", leave=True):
                raw_path = os.path.join(self.config.input_dir, file)
                save_path = os.path.join(self.config.simplified_dir, exp_name, file)

                if os.path.exists(save_path):
                    continue

                with open(raw_path, "r", encoding="utf-8") as f:
                    content_raw = f.read()

                start_time = time.time()
                content_simplified = self._simplify_md(content_raw, model, prompt_version)

                if content_simplified is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content_simplified)

                    with open(self.config.time_csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            model,
                            prompt_version,
                            file,
                            round(time.time() - start_time, 3),
                        ])
        print("  -> Step 1 完成。")

    def _check_existing_qa_files(self):
        """
        检查qa_source_dir目录是否存在且包含QA文件
        返回: True 如果存在可用文件, False 否则
        """
        if not os.path.exists(self.config.qa_source_dir):
            return False

        # 检查目录中是否有md文件
        existing_files = [f for f in os.listdir(self.config.qa_source_dir) if f.endswith(".md")]
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
            for file in simple_progress_bar(self.files, desc="    复制进度", leave=True):
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
            for file in simple_progress_bar(self.files, desc="    生成进度", leave=True):
                raw_path = os.path.join(self.config.input_dir, file)
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
            for file in simple_progress_bar(self.files, desc="    答题进度", leave=True):
                simplified_path = os.path.join(self.config.simplified_dir, exp_name, file)
                qa_gen_path = os.path.join(self.config.qa_gen_dir, file)
                save_path = os.path.join(self.config.qa_ans_dir, exp_name, file)

                if not os.path.exists(simplified_path) or not os.path.exists(qa_gen_path):
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

                answer_content = self._answer_qa_md(content_simplified, q_content, model_a)
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
            for file in simple_progress_bar(self.files, desc="    打分进度", leave=True):
                qa_ans_path = os.path.join(self.config.qa_ans_dir, exp_name, file)
                qa_gen_path = os.path.join(self.config.qa_gen_dir, file)
                save_path = os.path.join(self.config.qa_score_dir, exp_name, file)

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
            for file in self.files:
                score_file = os.path.join(score_dir, file)
                simplified_file = os.path.join(self.config.simplified_dir, exp_name, file)
                raw_file = os.path.join(self.config.input_dir, file)

                if not (os.path.exists(score_file) and os.path.exists(simplified_file) and os.path.exists(raw_file)):
                    continue

                with open(score_file, "r", encoding="utf-8") as f:
                    score_text = f.read()
                with open(raw_file, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                with open(simplified_file, "r", encoding="utf-8") as f:
                    simplified_text = f.read()

                parsed = self._parse_score_content(score_text)
                raw_len = self._count_chars(raw_text)
                compressed_len = self._count_chars(simplified_text)
                all_score = parsed["all_score"]
                max_score = parsed["max_score"]
                question_count = parsed["question_count"]
                omission_count = parsed["omission_count"]

                row = {
                    "file": file,
                    "model": model,
                    "prompt_version": prompt_version,
                    "experiment": exp_name,
                    "raw_len": raw_len,
                    "compressed_len": compressed_len,
                    "compression_ratio": round(compressed_len / raw_len, 6) if raw_len > 0 else None,
                    "all_score": all_score,
                    "max_score": max_score,
                    "question_count": question_count,
                    "score_rate": round(all_score / max_score, 6) if max_score > 0 else None,
                    "omission_count": omission_count,
                    "omission_rate": round(omission_count / question_count, 6) if question_count > 0 else None,
                }

                for q_type in QUESTION_TYPES:
                    type_score = parsed["type_score_map"][q_type]
                    type_count = parsed["type_count_map"][q_type]
                    type_max = type_count * 2
                    row[f"{q_type}_score"] = type_score
                    row[f"{q_type}_max"] = type_max
                    row[f"{q_type}_score_rate"] = (
                        round(type_score / type_max, 6) if type_max > 0 else None
                    )

                doc_rows.append(row)
                record_count += 1

            print(f"  -> 实验 {exp_name}: 成功统计 {record_count} 份记录。")

        if not doc_rows:
            print("  -> 未发现可汇总记录。")
            return

        df_doc = pd.DataFrame(doc_rows)
        df_doc.to_csv(self.config.doc_csv_path, index=False)

        agg_cols = {
            "file": "count",
            "score_rate": "mean",
            "compression_ratio": "mean",
            "omission_rate": "mean",
            "raw_len": "mean",
            "compressed_len": "mean",
            "all_score": "mean",
            "max_score": "mean",
            "global_score_rate": "mean",
            "entity_action_score_rate": "mean",
            "number_time_score_rate": "mean",
            "relation_score_rate": "mean",
            "conclusion_score_rate": "mean",
        }
        df_exp = (
            df_doc.groupby(["model", "prompt_version"], as_index=False)
            .agg(agg_cols)
            .rename(columns={
                "file": "docs_scored",
                "score_rate": "avg_score_rate",
                "compression_ratio": "avg_compression_ratio",
                "omission_rate": "avg_omission_rate",
                "raw_len": "avg_raw_len",
                "compressed_len": "avg_compressed_len",
                "all_score": "avg_all_score",
                "max_score": "avg_max_score",
                "global_score_rate": "avg_global_score_rate",
                "entity_action_score_rate": "avg_entity_action_score_rate",
                "number_time_score_rate": "avg_number_time_score_rate",
                "relation_score_rate": "avg_relation_score_rate",
                "conclusion_score_rate": "avg_conclusion_score_rate",
            })
        )
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

        if not self.files:
            print(f"[警告] 未在 {self.config.input_dir} 发现任何 .md 文件，流水线终止。")
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
