import json
import re
import argparse
from tqdm import tqdm
from collections import Counter

def extract_box_answer(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    return None

def normalize_math_str(s):
    if s is None:
        return None
    s = str(s)
    s = s.replace("\\,", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("$", "").replace("\\displaystyle", "").replace("\\text", "")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s

def is_equivalent(pred, gt):
    if normalize_math_str(pred) == normalize_math_str(gt):
        return True
    return False

def check_answer(response, gt_raw):
    # 从模型回答中提取 \boxed{}，若无则使用整个回答
    pred_boxed = extract_box_answer(response)
    pred_clean = pred_boxed.strip() if pred_boxed else response.strip()

    # 从标准答案中提取 \boxed{}，若无则直接使用
    gt_boxed = extract_box_answer(gt_raw)
    gt_clean = gt_boxed.strip() if gt_boxed else gt_raw.strip()

    return int(is_equivalent(pred_clean, gt_clean))

def main():
    parser = argparse.ArgumentParser(description="Check answers and score rollouts")
    parser.add_argument("--input", default="./output/rollout.jsonl",
                        help="Input rollout jsonl file")
    parser.add_argument("--output", default="./output/scored.jsonl",
                        help="Output scored jsonl file")
    args = parser.parse_args()

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} rollout records")

    correct_nums = []
    with open(args.output, "w", encoding="utf-8") as f:
        for item in tqdm(data):
            original = item["data"]
            gt = original.get("answer", "")
            correct = [check_answer(resp, gt) for resp in item["generations"]]
            item["correct"] = correct
            item["correct_num"] = sum(correct)
            correct_nums.append(item["correct_num"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Scoring completed. Saved to {args.output}")

    # 统计 correct_num 分布
    dist = Counter(correct_nums)
    print("\nCorrect_num distribution (over all samples):")
    for k in sorted(dist.keys()):
        print(f"  {k}/3: {dist[k]} samples ({dist[k]/len(correct_nums)*100:.1f}%)")

    # 打印前 3 条样本的 correct_num 供目视检查
    print("\nFirst 3 scored samples (correct_num):")
    for i, item in enumerate(data[:3]):
        print(f"  Sample {i+1}: correct_num = {item['correct_num']}, correct list = {item['correct']}")

if __name__ == "__main__":
    main()
