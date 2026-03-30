import os
import re
import sys
import time
import csv
import json
from datetime import datetime
from natsort import natsorted as na
import pandas as pd
from model import invoke_model_ds_rr


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
        self.compress_prompt_versions = ["baseline", "anchor_v2"]

        # 3. 评测角色模型（建议固定，减少评测抖动）
        self.baseline_model = "ali-kimi-k2.5"
        self.test_model = "ali-kimi-k2.5"
        self.judge_model = "ali-kimi-k2.5"

        # 4. 压缩长度配置
        self.compress_n = 500
        self.compress_m = 1000

        # 5. 输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_base_dir = f"eval_output_{self.timestamp}"
        self.simplified_dir = os.path.join(self.output_base_dir, "1_simplified_docs")
        self.baseline_task_dir = os.path.join(
            self.output_base_dir, "2_baseline_task_output"
        )
        self.test_task_dir = os.path.join(self.output_base_dir, "3_test_task_output")
        self.judge_score_dir = os.path.join(self.output_base_dir, "4_judge_scores")

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

        self.task_instructions = {
            "0_未分类或杂项": "提取文本的核心观点、关键人物/实体，并进行简洁的总结。",
            "1_长篇结构化应用文档": "撰写一份 500 字的执行摘要，要求提炼核心主旨、关键举措（谁做了什么）以及最终成果数据。",
            "2_学术论文与专业文献": "提取该研究的核心问题、实验方法、核心数据参数和结论，形成结构化图谱。",
            "3_二维表格与日程数据": "将表格信息转化为连贯的叙述性文本，说明所有的关键时间节点和对应事件，不能有时间错位。",
            "4_极短视觉描述": "重写一段场景描述，必须包含所有的视觉细节、主体特征和背景氛围。",
            "5_文学段落与扩写片段": "提取文本中的核心人物情绪变化轨迹、关键动作和环境描写要素。",
            "6_纯信息罗列清单": "按照类别将信息进行分组整理，并总结清单的核心范围。",
        }

    def _create_dirs(self):
        for d in [
            self.config.output_base_dir,
            self.config.simplified_dir,
            self.config.baseline_task_dir,
            self.config.test_task_dir,
            self.config.judge_score_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        for category in self.config.categories:
            os.makedirs(
                os.path.join(self.config.baseline_task_dir, category), exist_ok=True
            )

        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            os.makedirs(
                os.path.join(self.config.simplified_dir, exp_name), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.config.test_task_dir, exp_name), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.config.judge_score_dir, exp_name), exist_ok=True
            )
            # 为每个实验创建类别子目录
            for category in self.config.categories:
                os.makedirs(
                    os.path.join(self.config.simplified_dir, exp_name, category),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(self.config.test_task_dir, exp_name, category),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(self.config.judge_score_dir, exp_name, category),
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

        raise ValueError(f"未知的 prompt_version: {prompt_version}")

    def _simplify_md(self, content, model_type, prompt_version):
        max_nums = self._get_max_nums(content)
        if max_nums is None:
            return None
        prompt = self._build_simplify_prompt(content, max_nums, prompt_version)
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _build_task_prompt(self, task_instruction, content):
        return f"""# Role
你是一个极高素养的文档分析与写作专家。
# Task
请基于提供的【参考文本】，完成以下生成任务：
【{task_instruction}】
# Rules
1. 绝对忠于参考文本，严禁引入外部知识或脑补细节。
2. 保持逻辑严密，结构清晰。如果参考文本中缺乏某些信息，请在生成时自然略过，不要自行虚构。
3. 字数控制在 300 - 500 字以内。直接输出任务结果，不要包含任何如“好的，这是报告”之类的开场白。
# 参考文本
{content}
"""

    def _generate_task_output(self, content, category, model_type):
        task_instruction = self.task_instructions.get(
            category, self.task_instructions["0_未分类或杂项"]
        )
        prompt = self._build_task_prompt(task_instruction, content)
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _build_judge_prompt(self, baseline_content, test_content):
        return f"""# Role
你是一个极其严苛且精准的文本评估裁判。你需要对比评估一份【测试生成文档】与【基准金标文档】之间的吻合度。
# Background
【基准金标文档】是基于完整原始数据生成的。
【测试生成文档】是基于经过“有损压缩后的数据”生成的。
你的任务是评估：由于数据压缩，测试文档在生成时丢失了多少核心信息？是否出现了逻辑错误或幻觉？
# 输入数据
[基准金标文档 (Baseline)]：
{baseline_content}
[测试生成文档 (Test)]：
{test_content}
# 评分标准 (满分10分)
请从以下三个维度独立打分：
1. 核心信息完备度 (0-5分)：
   - 5分：测试文档完美保留了基准文档中的所有核心实体、关键数据、核心事件。
   - 3分：保留了主干，但丢失了部分重要细节或修饰性限定词。
   - 1分：丢失了严重的核心事实，只保留了极其粗略的框架。
   - 0分：信息面目全非，几乎没有有效信息重合。
2. 逻辑准确性与幻觉 (0-3分)：
   - 3分：逻辑关系完全正确，没有张冠李戴，没有捏造信息（幻觉）。
   - 1.5分：存在轻微的逻辑混淆或人物/事件对应错误。
   - 0分：存在严重的逻辑错乱或大面积幻觉（自行编造了基准文档中不存在的事实）。
3. 生成质量与行文流畅度 (0-2分)：
   - 2分：行文流畅，结构合理，与基准文档水平相当。
   - 1分：行文生硬，像是在机械罗列残缺的短语。
   - 0分：句子不通顺，排版彻底崩坏。
# 输出格式限制
必须严格按照以下 JSON 格式输出，不要包含任何 markdown 格式化标记(```json)或其他说明文字：
{{
  "completeness_score": X,
  "completeness_reason": "简短理由",
  "accuracy_score": Y,
  "accuracy_reason": "简短理由",
  "quality_score": Z,
  "quality_reason": "简短理由",
  "total_score": W
}}
"""

    def _score_generation(self, baseline_content, test_content, model_type):
        time.sleep(1)
        prompt = self._build_judge_prompt(baseline_content, test_content)
        result = invoke_model_ds_rr(prompt, model=model_type)
        return result if isinstance(result, str) else None

    @staticmethod
    def _parse_judge_score(score_text):
        try:
            # 尝试清理可能带有的 markdown code block 标记
            clean_text = score_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]

            clean_text = clean_text.strip()
            data = json.loads(clean_text)

            # 确保包含了所有必要的字段
            expected_keys = [
                "completeness_score",
                "accuracy_score",
                "quality_score",
                "total_score",
            ]
            for key in expected_keys:
                if key not in data:
                    data[key] = 0.0  # Fallback

            return data
        except json.JSONDecodeError:
            # 解析失败时的回退机制，尝试正则匹配
            data = {
                "completeness_score": 0.0,
                "accuracy_score": 0.0,
                "quality_score": 0.0,
                "total_score": 0.0,
            }
            comp_match = re.search(r'"completeness_score"\s*:\s*([\d\.]+)', score_text)
            if comp_match:
                data["completeness_score"] = float(comp_match.group(1))

            acc_match = re.search(r'"accuracy_score"\s*:\s*([\d\.]+)', score_text)
            if acc_match:
                data["accuracy_score"] = float(acc_match.group(1))

            qual_match = re.search(r'"quality_score"\s*:\s*([\d\.]+)', score_text)
            if qual_match:
                data["quality_score"] = float(qual_match.group(1))

            tot_match = re.search(r'"total_score"\s*:\s*([\d\.]+)', score_text)
            if tot_match:
                data["total_score"] = float(tot_match.group(1))

            return data

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

    def run_step_2_generate_baseline(self):
        print(f"\n[Step 2/5] 基于原始文档生成基准任务答案 (Baseline)...")
        model = self.config.baseline_model
        print(f"  -> Baseline 生成模型: {model}")

        for category, file in simple_progress_bar(
            self.all_files, desc="    生成进度", leave=True
        ):
            raw_path = os.path.join(self.config.input_dir, category, file)
            save_path = os.path.join(self.config.baseline_task_dir, category, file)

            if os.path.exists(save_path):
                continue

            with open(raw_path, "r", encoding="utf-8") as f:
                content_raw = f.read()

            task_output = self._generate_task_output(content_raw, category, model)
            if task_output is not None:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(task_output)
        print("  -> Step 2 完成。")

    def run_step_3_generate_test(self):
        print(f"\n[Step 3/5] 基于压缩文档生成测试任务答案 (Test Output)...")
        model = self.config.test_model
        print(f"  -> Test 生成模型: {model}")
        for compress_model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(compress_model, prompt_version)
            print(f"  -> 处理 {exp_name} 的压缩结果")
            for category, file in simple_progress_bar(
                self.all_files, desc="    答题进度", leave=True
            ):
                simplified_path = os.path.join(
                    self.config.simplified_dir, exp_name, category, file
                )
                save_path = os.path.join(
                    self.config.test_task_dir, exp_name, category, file
                )

                if not os.path.exists(simplified_path):
                    continue
                if os.path.exists(save_path):
                    continue

                with open(simplified_path, "r", encoding="utf-8") as f:
                    content_simplified = f.read()

                task_output = self._generate_task_output(
                    content_simplified, category, model
                )
                if task_output is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(task_output)
        print("  -> Step 3 完成。")

    def run_step_4_score_generation(self):
        print(f"\n[Step 4/5] 裁判对比基准与测试输出打分...")
        model_s = self.config.judge_model
        print(f"  -> 裁判打分模型: {model_s}")
        for compress_model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(compress_model, prompt_version)
            print(f"  -> 评估 {exp_name} 的生成结果")
            for category, file in simple_progress_bar(
                self.all_files, desc="    打分进度", leave=True
            ):
                baseline_path = os.path.join(
                    self.config.baseline_task_dir, category, file
                )
                test_path = os.path.join(
                    self.config.test_task_dir, exp_name, category, file
                )
                save_path = os.path.join(
                    self.config.judge_score_dir, exp_name, category, file
                )

                if not os.path.exists(baseline_path) or not os.path.exists(test_path):
                    continue
                if os.path.exists(save_path):
                    continue

                with open(baseline_path, "r", encoding="utf-8") as f:
                    content_baseline = f.read()
                with open(test_path, "r", encoding="utf-8") as f:
                    content_test = f.read()

                score_content = self._score_generation(
                    content_baseline, content_test, model_s
                )
                if score_content is not None:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(score_content)
        print("  -> Step 4 完成。")

    def run_step_5_aggregate_scores(self):
        print(f"\n[Step 5/5] 汇总分数并生成统计报表...")
        doc_rows = []

        for model, prompt_version in self.config.experiment_keys():
            exp_name = self._exp_name(model, prompt_version)
            score_dir = os.path.join(self.config.judge_score_dir, exp_name)
            if not os.path.exists(score_dir):
                continue

            record_count = 0
            for category, file in self.all_files:
                score_file = os.path.join(score_dir, category, file)

                if not os.path.exists(score_file):
                    continue

                with open(score_file, "r", encoding="utf-8") as f:
                    score_text = f.read()

                parsed = self._parse_judge_score(score_text)

                row = {
                    "file": file,
                    "category": category,
                    "model": model,
                    "prompt_version": prompt_version,
                    "completeness_score": parsed.get("completeness_score", 0.0),
                    "accuracy_score": parsed.get("accuracy_score", 0.0),
                    "quality_score": parsed.get("quality_score", 0.0),
                    "total_score": parsed.get("total_score", 0.0),
                }

                doc_rows.append(row)
                record_count += 1

            print(f"  -> 实验 {exp_name}: 成功统计 {record_count} 份记录。")

        if not doc_rows:
            print("  -> 未发现可汇总记录。")
            return

        df_doc = pd.DataFrame(doc_rows)
        df_doc.to_csv(self.config.doc_csv_path, index=False)

        # 实验级汇总
        df_exp = (
            df_doc.groupby(["model", "prompt_version"], as_index=False)
            .agg(
                {
                    "completeness_score": "mean",
                    "accuracy_score": "mean",
                    "quality_score": "mean",
                    "total_score": "mean",
                }
            )
            .rename(
                columns={
                    "model": "模型",
                    "prompt_version": "Prompt版本",
                    "completeness_score": "总体完备度得分",
                    "accuracy_score": "总体准确度得分",
                    "quality_score": "总体生成质量得分",
                    "total_score": "总体平均总分",
                }
            )
        )

        # 计算各类文档平均得分
        df_cat = df_doc.groupby(
            ["model", "prompt_version", "category"], as_index=False
        ).agg(
            {
                "completeness_score": "mean",
                "accuracy_score": "mean",
                "quality_score": "mean",
                "total_score": "mean",
            }
        )

        for category in self.config.categories:
            cat_data = df_cat[df_cat["category"] == category]
            if len(cat_data) > 0:
                for _, row in cat_data.iterrows():
                    mask = (df_exp["模型"] == row["model"]) & (
                        df_exp["Prompt版本"] == row["prompt_version"]
                    )
                    if mask.any():
                        df_exp.loc[mask, f"{category}_完备度"] = row[
                            "completeness_score"
                        ]
                        df_exp.loc[mask, f"{category}_准确度"] = row["accuracy_score"]
                        df_exp.loc[mask, f"{category}_生成质量"] = row["quality_score"]
                        df_exp.loc[mask, f"{category}_总分"] = row["total_score"]

        df_exp.to_csv(self.config.exp_csv_path, index=False)

        print(f"  -> 文档级统计 CSV 已输出至: {self.config.doc_csv_path}")
        print(f"  -> 实验级统计 CSV 已输出至: {self.config.exp_csv_path}")
        print("  -> Step 5 完成。")

    def run_all(self):
        print("===========================================================")
        print("  LLM 任务导向评测 (Task-Oriented Eval) 流水线")
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
        self.run_step_2_generate_baseline()
        self.run_step_3_generate_test()
        self.run_step_4_score_generation()
        self.run_step_5_aggregate_scores()

        print("\n===========================================================")
        print("  🎉 流水线执行完毕！")
        print(f"  📂 所有数据保存在: {self.config.output_base_dir}")
        print("===========================================================\n")


if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = DocumentEvaluationPipeline(config)
    pipeline.run_all()
