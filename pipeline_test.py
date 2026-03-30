import os
import re
import sys
import time
import csv
import json
import random
from datetime import datetime
from natsort import natsorted as na
import pandas as pd
USE_MOCK_MODE = True 

def invoke_llm(prompt, model="deepseek-v3"):
    if USE_MOCK_MODE:
        return mock_invoke(prompt, model)
    else:
        return invoke_model_ds_rr(prompt, model=model)


def simple_progress_bar(iterable, desc="", leave=True):
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
        self.input_dir = "/home/users/chenxi84/chenxi/new_pipeline/categorized_files"

        # 类别映射
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
        self.compress_prompt_versions = ["baseline", "anchor_v3"]

        # 3. 评测角色模型
        self.baseline_model = "deepseek-v3"
        self.test_model = "deepseek-v3"
        self.judge_model = "deepseek-v3"

        # 4. 压缩长度配置
        self.compress_n = 500
        self.compress_m = 1000

        # 5. 测试样本数量
        self.test_sample_size = 30

        # 6. 输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_base_dir = f"eval_output_test_{self.timestamp}"
        self.simplified_dir = os.path.join(self.output_base_dir, "1_simplified_docs")
        self.baseline_task_dir = os.path.join(
            self.output_base_dir, "2_baseline_task_output"
        )
        self.test_task_dir = os.path.join(self.output_base_dir, "3_test_task_output")
        self.judge_score_dir = os.path.join(self.output_base_dir, "4_judge_scores")

        # 7. 统计文件
        models_str = "_".join(self.models_to_test)
        self.doc_csv_path = os.path.join(
            self.output_base_dir, f"{models_str}_scores_summary.csv"
        )
        self.exp_csv_path = os.path.join(
            self.output_base_dir, f"{models_str}_experiment_summary.csv"
        )

    def experiment_keys(self):
        for model in self.models_to_test:
            for prompt_version in self.compress_prompt_versions:
                yield model, prompt_version


class DocumentEvaluationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._create_dirs()
        self.files_by_category = {}
        self.all_files = []

        if os.path.exists(self.config.input_dir):
            for category in self.config.categories:
                category_path = os.path.join(self.config.input_dir, category)
                if os.path.exists(category_path):
                    files = [
                        f for f in na(os.listdir(category_path)) if f.endswith(".md")
                    ]
                    if files:
                        self.files_by_category[category] = files
                        for f in files:
                            self.all_files.append((category, f))

        # 随机抽取 30 个样本
        random.seed(42)
        if len(self.all_files) > self.config.test_sample_size:
            self.all_files = random.sample(self.all_files, self.config.test_sample_size)

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

        for model, prompt_version in self.config.experiment_keys():
            os.makedirs(
                os.path.join(self.config.simplified_dir, f"{model}__{prompt_version}"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(self.config.baseline_task_dir, model), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.config.test_task_dir, f"{model}__{prompt_version}"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(self.config.judge_score_dir, f"{model}__{prompt_version}"),
                exist_ok=True,
            )

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
            return f"""# Role
你是一个极高素养的文档压缩专家。你的任务是在**保证事实完整性**的前提下，对文档进行精简。

# ⚡ 核心原则
1. **保真优先**：压缩是对信息的提炼，不是重构。严禁在输出中引入原文未提及的事实、数据、概念。
2. **推理但不臆造**：允许基于原文信息进行合理推断（如从上下文推断事件目的），但禁止捏造具体数据或细节。
3. **可回溯核查**：你输出的每一个事实性陈述，都必须能在原文找到对应。

# 自适应压缩策略（根据文档类型选择）
1. 【学术论文/专业文献】：必须保留：研究目的、核心方法、关键数据参数、核心结论。允许精简：文献综述背景、详细实验过程描述。
2. 【多篇集合与并列项】（如多份总结/合集）：必须保留：每个独立篇目的核心事件、身份、结果。禁止将多个主体的事件揉捏成一个统一叙述。
3. 【高密度数据与映射关系】（如生字表、课表）：必须保留：原有的分组/层级映射结构。组内可使用紧凑逗号分隔，但严禁将不同分组的数据拍平混淆。
4. 【结构化模板与大纲】（如工作总结）：必须保留：每个模块的维度划分、核心举措、成果数据。禁止为了压缩而使用斜杠生硬拼接。
5. 【一般连续性文本】（普通故事/文章）：剔除客套话和修饰词，使用客观短句。

# 字数与底线约束
- 目标限制：压缩到 {max_nums} 字符以内，优先保证信息完整。
- 硬性保真：以下信息绝对不能丢失——核心数据（如百分比、金额）、专有名词、关键人物/地点/事件。

# 待压缩文本
{content}

# 输出要求
请直接输出压缩后的文本。确保每个事实性陈述都可回溯至原文。
"""

        raise ValueError(f"未知的 prompt_version: {prompt_version}")

    def _simplify_md(self, content, model_type, prompt_version):
        max_nums = self._get_max_nums(content)
        if max_nums is None:
            return None
        prompt = self._build_simplify_prompt(content, max_nums, prompt_version)
        result = invoke_llm(prompt, model=model_type)
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
3. 字数控制在 300 - 500 字以内。直接输出任务结果，不要包含任何如"好的，这是报告"之类的开场白。
# 参考文本
{content}
"""

    def _generate_task_output(self, content, category, model_type):
        task_instruction = self.task_instructions.get(
            category, self.task_instructions["0_未分类或杂项"]
        )
        prompt = self._build_task_prompt(task_instruction, content)
        result = invoke_llm(prompt, model=model_type)
        return result if isinstance(result, str) else None

    def _build_judge_prompt(self, baseline_content, test_content):
        return f"""# Role
你是一个极其严苛且精准的文本评估裁判。你需要对比评估一份【测试生成文档】与【基准金标文档】之间的吻合度。
# Background
【基准金标文档】是基于完整原始数据生成的。
【测试生成文档】是基于经过"有损压缩后的数据"生成的。
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
        prompt = self._build_judge_prompt(baseline_content, test_content)
        result = invoke_llm(prompt, model=model_type)
        if not isinstance(result, str):
            return None
        try:
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            score_data = json.loads(result)
            return {
                "completeness_score": float(score_data.get("completeness_score", 0)),
                "accuracy_score": float(score_data.get("accuracy_score", 0)),
                "quality_score": float(score_data.get("quality_score", 0)),
                "total_score": float(score_data.get("total_score", 0)),
                "reason": f"{score_data.get('completeness_reason', '')} | {score_data.get('accuracy_reason', '')} | {score_data.get('quality_reason', '')}",
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n[!] JSON 解析失败: {e}")
            return None

    def run_step_1_simplify(self):
        print(f"\n[Step 1/4] 开始压缩文档 (共 {len(self.all_files)} 篇)...")
        for model, prompt_version in simple_progress_bar(
            list(self.config.experiment_keys()), desc="    实验组合", leave=True
        ):
            print(f"\n  -> 模型: {model}, Prompt: {prompt_version}")
            for category, file in simple_progress_bar(
                self.all_files, desc=f"    压缩进度", leave=False
            ):
                raw_path = os.path.join(self.config.input_dir, category, file)
                save_path = os.path.join(
                    self.config.simplified_dir,
                    f"{model}__{prompt_version}",
                    category,
                    file,
                )

                if os.path.exists(save_path):
                    continue

                with open(raw_path, "r", encoding="utf-8") as f:
                    content_raw = f.read()

                content_simplified = self._simplify_md(
                    content_raw, model, prompt_version
                )

                if content_simplified is not None:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(content_simplified)
        print("\n  -> Step 1 完成。")

    def run_step_2_generate_baseline(self):
        print(f"\n[Step 2/4] 基于原始文档生成基准任务输出...")
        model = self.config.baseline_model
        for category, file in simple_progress_bar(
            self.all_files, desc="    基准生成", leave=True
        ):
            raw_path = os.path.join(self.config.input_dir, category, file)
            save_path = os.path.join(
                self.config.baseline_task_dir, model, category, file
            )

            if os.path.exists(save_path):
                continue

            with open(raw_path, "r", encoding="utf-8") as f:
                content_raw = f.read()

            task_output = self._generate_task_output(content_raw, category, model)
            if task_output is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(task_output)
        print("  -> Step 2 完成。")

    def run_step_3_generate_test_and_score(self):
        print(f"\n[Step 3/4] 基于压缩文档生成测试任务并进行评分...")
        model_t = self.config.test_model
        model_s = self.config.judge_model

        for model, prompt_version in simple_progress_bar(
            list(self.config.experiment_keys()), desc="    实验组合", leave=True
        ):
            print(f"\n  -> 评测: {model} + {prompt_version}")
            for category, file in simple_progress_bar(
                self.all_files, desc="    测试评分", leave=False
            ):
                simplified_path = os.path.join(
                    self.config.simplified_dir,
                    f"{model}__{prompt_version}",
                    category,
                    file,
                )
                baseline_path = os.path.join(
                    self.config.baseline_task_dir, model_t, category, file
                )
                test_save_path = os.path.join(
                    self.config.test_task_dir,
                    f"{model}__{prompt_version}",
                    category,
                    file,
                )
                score_save_path = os.path.join(
                    self.config.judge_score_dir,
                    f"{model}__{prompt_version}",
                    category,
                    file,
                )

                if not os.path.exists(simplified_path) or not os.path.exists(
                    baseline_path
                ):
                    continue
                if os.path.exists(score_save_path):
                    continue

                with open(simplified_path, "r", encoding="utf-8") as f:
                    content_simplified = f.read()
                with open(baseline_path, "r", encoding="utf-8") as f:
                    content_baseline = f.read()

                task_output = self._generate_task_output(
                    content_simplified, category, model_t
                )
                if task_output is not None:
                    os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
                    with open(test_save_path, "w", encoding="utf-8") as f:
                        f.write(task_output)

                    score_data = self._score_generation(
                        content_baseline, task_output, model_s
                    )
                    if score_data is not None:
                        os.makedirs(os.path.dirname(score_save_path), exist_ok=True)
                        with open(score_save_path, "w", encoding="utf-8") as f:
                            json.dump(score_data, f, ensure_ascii=False, indent=2)
        print("\n  -> Step 3 完成。")

    def run_step_4_aggregate_scores(self):
        print(f"\n[Step 4/4] 汇总分数并生成统计报表...")
        all_data_score = {}

        for model, prompt_version in self.config.experiment_keys():
            score_dir = os.path.join(
                self.config.judge_score_dir, f"{model}__{prompt_version}"
            )
            if not os.path.exists(score_dir):
                continue

            completeness_scores = []
            accuracy_scores = []
            quality_scores = []
            total_scores = []

            for category in self.config.categories:
                category_score_dir = os.path.join(score_dir, category)
                if not os.path.exists(category_score_dir):
                    continue

                for file in os.listdir(category_score_dir):
                    if not file.endswith(".md"):
                        continue
                    score_file = os.path.join(category_score_dir, file)
                    try:
                        with open(score_file, "r", encoding="utf-8") as f:
                            score_data = json.load(f)
                            completeness_scores.append(
                                score_data.get("completeness_score", 0)
                            )
                            accuracy_scores.append(score_data.get("accuracy_score", 0))
                            quality_scores.append(score_data.get("quality_score", 0))
                            total_scores.append(score_data.get("total_score", 0))
                    except:
                        pass

            if len(total_scores) > 0:
                key = f"{model}_{prompt_version}"
                all_data_score[f"{key}_completeness"] = completeness_scores
                all_data_score[f"{key}_accuracy"] = accuracy_scores
                all_data_score[f"{key}_quality"] = quality_scores
                all_data_score[f"{key}_total"] = total_scores

        if all_data_score:
            df = pd.DataFrame(all_data_score)
            df.to_csv(self.config.doc_csv_path, index=False)

            summary_data = {}
            for model, prompt_version in self.config.experiment_keys():
                key = f"{model}_{prompt_version}"
                total_scores = all_data_score.get(f"{key}_total", [])
                completeness = all_data_score.get(f"{key}_completeness", [])
                accuracy = all_data_score.get(f"{key}_accuracy", [])
                quality = all_data_score.get(f"{key}_quality", [])

                if total_scores:
                    summary_data[f"{model}_{prompt_version}_总体完备度"] = [
                        sum(completeness) / len(completeness) if completeness else 0
                    ]
                    summary_data[f"{model}_{prompt_version}_总体准确度"] = [
                        sum(accuracy) / len(accuracy) if accuracy else 0
                    ]
                    summary_data[f"{model}_{prompt_version}_总体生成质量"] = [
                        sum(quality) / len(quality) if quality else 0
                    ]
                    summary_data[f"{model}_{prompt_version}_总体平均总分"] = [
                        sum(total_scores) / len(total_scores) if total_scores else 0
                    ]

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.config.exp_csv_path, index=False)
            print(f"  -> 详细分数: {self.config.doc_csv_path}")
            print(f"  -> 汇总结果: {self.config.exp_csv_path}")

        print("  -> Step 4 完成。")

    def run_all(self):
        print(f"===========================================================")
        print(f"  LLM 文本压缩评估流水线 (测试版 - 30 样本)")
        print(f"  启动时间: {self.config.timestamp}")
        print(f"  输入目录: {self.config.input_dir}")
        print(f"  测试样本数: {len(self.all_files)}")
        print(f"  输出目录: {self.config.output_base_dir}")
        print(f"  实验组合: {[f'{m}+{p}' for m, p in self.config.experiment_keys()]}")
        print(f"===========================================================\n")

        if not self.all_files:
            print(f"[警告] 未发现任何测试文件，流水线终止。")
            return

        self.run_step_1_simplify()
        self.run_step_2_generate_baseline()
        self.run_step_3_generate_test_and_score()
        self.run_step_4_aggregate_scores()

        print(f"\n===========================================================")
        print(f"  🎉 流水线执行完毕！")
        print(f"  📂 所有数据保存在: {self.config.output_base_dir}")
        print(f"===========================================================\n")


if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = DocumentEvaluationPipeline(config)
    pipeline.run_all()
