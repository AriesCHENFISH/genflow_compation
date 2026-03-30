import os
import re
import time
import csv
from datetime import datetime
from natsort import natsorted as na
from model import invoke_model_ds_rr

# ==================== 配置 ====================
INPUT_DIR = "/home/users/chenxi84/chenxi/new_pipeline/categorized_files_real"
CATEGORIES = [
    "0_未分类或杂项",
    "1_长篇结构化应用文档",
    "2_学术论文与专业文献",
    "3_二维表格与日程数据",
    "4_极短视觉描述",
    "5_文学段落与扩写片段",
    "6_纯信息罗列清单",
]
MODEL_COMPRESS = "qwen3.5-27b"          # 压缩模型
QA_GEN_MODEL = "deepseek-v3.2"          # QA 生成模型
QA_ANS_MODEL = "deepseek-v3.2"          # QA 回答模型
QA_SCORE_MODEL = "deepseek-v3.2"        # 评分模型

COMPRESS_N = 500
COMPRESS_M = 1000
SAMPLES_PER_CATEGORY = 2                # 每个类别抽取文档数（总文档数约为 len(CATEGORIES)*SAMPLES_PER_CATEGORY）
OUTPUT_DIR = "prompt_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# 问题类型列表（用于分数统计）
QUESTION_TYPES = ["global", "entity_action", "number_time", "relation", "conclusion"]

# ==================== 辅助函数 ====================
def get_max_nums(content):
    n, m = COMPRESS_N, COMPRESS_M
    if len(content) > n and len(content) <= m:
        return int(len(content) * 0.2)
    elif len(content) > m:
        return 500
    else:
        return None

def build_prompt_anchor_v3(content, max_nums):
    return f"""【最高指令：极速直出，禁止任何思考、分析或开场白，首字必须是正文！】

你是一个零延迟的机械文本压缩器。请将下文压缩至 **{max_nums} 字符以内**。为追求极致速度与保真，请机械执行以下操作，**严禁进行复杂的归纳、提炼或重写**：

1. 【锁死骨架】：严格保留原文的边界锚点（如"篇1："、"第X页："、"一、目的"、"时间："等）。**绝对禁止**将多篇独立文章融合成一篇，也**绝对禁止**将分段的文献压成不分段的实心砖。原文分了几块，你就输出几块。
2. 【剔肉留筋】：在每个骨架内部，直接机械删去修饰词、举例、客套话和长句过渡。仅保留：主体、核心动作、关键数据（数字/专有名词/结果）。使用极简但连贯的短句，**禁止**用大量斜杠（/）生硬拼接碎词。
3. 【数据降维】：遇到表格或密集列表，在保留其归属锚点的前提下，直接转化为紧凑的逗号分隔文本。禁用 Markdown 表格。

底线约束：绝对不超过 {max_nums} 字符；绝不改变原文的客观陈列形式；绝不遗漏核心数据。

待压缩文本：
{content}"""

def build_prompt_anchor_v3_fast(content, max_nums):
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

def generate_qa(content, model):
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
    return invoke_model_ds_rr(prompt, model=model)

def extract_questions_only(content):
    """从 QA 文本中仅提取问题"""
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

def answer_qa(content, q_content, model):
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
    return invoke_model_ds_rr(prompt, model=model)

def score_qa(reference_qa, answer_qa, model):
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
    return invoke_model_ds_rr(prompt, model=model)

def parse_score_content(score_text):
    q_pattern = re.compile(
        r"^Q(\d+)\s*\[type=([a-z_]+)\]\s*:\s*([012])\s*\|\s*([a-z_]+)",
        re.IGNORECASE,
    )
    type_score_map = {q_type: 0 for q_type in QUESTION_TYPES}
    type_count_map = {q_type: 0 for q_type in QUESTION_TYPES}
    question_count = 0
    omission_count = 0

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

    all_score = sum(type_score_map.values())
    max_score = question_count * 2
    return {
        "all_score": all_score,
        "max_score": max_score,
        "question_count": question_count,
        "omission_count": omission_count,
        "type_score_map": type_score_map,
        "type_count_map": type_count_map,
    }

# ==================== 核心流程 ====================
def get_sample_files():
    samples = []
    for cat in CATEGORIES:
        cat_path = os.path.join(INPUT_DIR, cat)
        if not os.path.isdir(cat_path):
            continue
        files = [f for f in na(os.listdir(cat_path)) if f.endswith(".md")]
        chosen = files[:SAMPLES_PER_CATEGORY]
        for f in chosen:
            samples.append((cat, os.path.join(cat_path, f), f))
    return samples

def compress(content, prompt_version):
    max_nums = get_max_nums(content)
    if max_nums is None:
        return None, None
    if prompt_version == "anchor_v3":
        prompt = build_prompt_anchor_v3(content, max_nums)
    elif prompt_version == "anchor_v3_fast":
        prompt = build_prompt_anchor_v3_fast(content, max_nums)
    else:
        raise ValueError("未知 prompt 版本")
    start = time.time()
    result = invoke_model_ds_rr(prompt, model=MODEL_COMPRESS)
    elapsed = time.time() - start
    return result, elapsed

def evaluate_single_file(original_path, original_content, compress_version, compress_func):
    """
    对单个文档执行压缩、QA 生成、回答、评分。
    compress_version: 'anchor_v3' or 'anchor_v3_fast'
    compress_func: 用于压缩的函数
    """
    # 1. 压缩
    compressed, comp_time = compress_func(original_content, compress_version)
    if compressed is None:
        return None  # 无需压缩，跳过

    # 2. 生成 QA（只做一次，但这里每个压缩版本都做一次，因为需要参考答案）
    # 实际上参考答案是同一个，我们可以缓存，但为了代码简单，每次都生成
    qa_ref = generate_qa(original_content, QA_GEN_MODEL)
    if not qa_ref:
        return None

    # 3. 回答问题（基于压缩文档）
    questions_only = extract_questions_only(qa_ref)
    if not questions_only:
        return None
    answer = answer_qa(compressed, questions_only, QA_ANS_MODEL)
    if not answer:
        return None

    # 4. 评分
    score_text = score_qa(qa_ref, answer, QA_SCORE_MODEL)
    if not score_text:
        return None
    parsed = parse_score_content(score_text)
    return {
        "compress_version": compress_version,
        "compressed_len": len(compressed),
        "compress_time": comp_time,
        "score_rate": parsed["all_score"] / parsed["max_score"] if parsed["max_score"] > 0 else None,
        "omission_rate": parsed["omission_count"] / parsed["question_count"] if parsed["question_count"] > 0 else None,
        "type_score_map": parsed["type_score_map"],
        "type_count_map": parsed["type_count_map"],
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = get_sample_files()
    if not samples:
        print("未找到任何文档，请检查输入目录。")
        return

    print(f"共抽取 {len(samples)} 篇文档进行对比")
    results = []  # 每篇文档的结果
    # 用于缓存 QA 生成的结果（避免重复生成）
    qa_cache = {}

    for idx, (cat, full_path, fname) in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] 处理 {cat}/{fname} ...")
        with open(full_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # 检查是否需要压缩
        max_nums = get_max_nums(original_content)
        if max_nums is None:
            print(f"  -> 文档长度 {len(original_content)} ≤ {COMPRESS_N}，无需压缩，跳过")
            continue

        # 为两个版本分别压缩和评测
        for ver in ["anchor_v3", "anchor_v3_fast"]:
            print(f"  -> 评测版本: {ver}")
            # 压缩
            compressed, comp_time = compress(original_content, ver)
            if compressed is None:
                print(f"    压缩失败或无需压缩")
                continue

            # 获取或生成 QA
            if fname not in qa_cache:
                print("    生成 QA...")
                qa_ref = generate_qa(original_content, QA_GEN_MODEL)
                qa_cache[fname] = qa_ref
            else:
                qa_ref = qa_cache[fname]
            if not qa_ref:
                print("    QA 生成失败")
                continue

            questions_only = extract_questions_only(qa_ref)
            if not questions_only:
                print("    提取问题失败")
                continue

            print("    回答 QA...")
            answer = answer_qa(compressed, questions_only, QA_ANS_MODEL)
            if not answer:
                print("    回答失败")
                continue

            print("    评分...")
            score_text = score_qa(qa_ref, answer, QA_SCORE_MODEL)
            if not score_text:
                print("    评分失败")
                continue

            parsed = parse_score_content(score_text)
            if parsed["max_score"] == 0:
                print("    评分解析失败")
                continue

            # 保存中间文件（可选）
            out_dir = os.path.join(OUTPUT_DIR, ver, cat)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                f.write(compressed)

            results.append({
                "category": cat,
                "file": fname,
                "version": ver,
                "original_len": len(original_content),
                "compressed_len": len(compressed),
                "compress_time": round(comp_time, 3),
                "score_rate": round(parsed["all_score"] / parsed["max_score"], 6),
                "omission_rate": round(parsed["omission_count"] / parsed["question_count"], 6),
                "type_scores": parsed["type_score_map"],
                "type_counts": parsed["type_count_map"],
            })

    # 保存结果 CSV
    csv_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "file", "version", "original_len", "compressed_len",
                         "compress_time", "score_rate", "omission_rate"])
        for r in results:
            writer.writerow([r["category"], r["file"], r["version"], r["original_len"],
                             r["compressed_len"], r["compress_time"], r["score_rate"], r["omission_rate"]])

    # 打印摘要
    print("\n===== 对比摘要 =====")
    # 按版本汇总
    for ver in ["anchor_v3", "anchor_v3_fast"]:
        ver_results = [r for r in results if r["version"] == ver]
        if not ver_results:
            continue
        total_time = sum(r["compress_time"] for r in ver_results)
        avg_time = total_time / len(ver_results)
        avg_score = sum(r["score_rate"] for r in ver_results) / len(ver_results)
        avg_omission = sum(r["omission_rate"] for r in ver_results) / len(ver_results)
        print(f"\n{ver}:")
        print(f"  文档数: {len(ver_results)}")
        print(f"  平均耗时: {avg_time:.2f} s/篇")
        print(f"  平均得分率: {avg_score:.4f}")
        print(f"  平均缺失率: {avg_omission:.4f}")

    # 如果两个版本都有结果，计算速度提升和得分率差异
    v3_results = [r for r in results if r["version"] == "anchor_v3"]
    fast_results = [r for r in results if r["version"] == "anchor_v3_fast"]
    if v3_results and fast_results:
        v3_avg_time = sum(r["compress_time"] for r in v3_results) / len(v3_results)
        fast_avg_time = sum(r["compress_time"] for r in fast_results) / len(fast_results)
        v3_avg_score = sum(r["score_rate"] for r in v3_results) / len(v3_results)
        fast_avg_score = sum(r["score_rate"] for r in fast_results) / len(fast_results)
        print("\n===== 速度与效果对比 =====")
        print(f"速度提升: {(v3_avg_time - fast_avg_time)/v3_avg_time*100:.1f}%")
        print(f"得分率变化: {fast_avg_score - v3_avg_score:+.4f}")

    print(f"\n详细结果已保存至: {csv_path}")
    print(f"压缩文件保存在: {OUTPUT_DIR}/anchor_v3/ 和 {OUTPUT_DIR}/anchor_v3_fast/")

if __name__ == "__main__":
    main()