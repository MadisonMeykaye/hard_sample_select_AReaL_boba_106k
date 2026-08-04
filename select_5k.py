import json
import random
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Select data for GRPO training")
    parser.add_argument("--input", default="./output/scored.jsonl",
                        help="Input scored jsonl file")
    parser.add_argument("--output", default="./output/grpo_train_5k.jsonl",
                        help="Output training jsonl file")
    parser.add_argument("--target", type=int, default=5000,
                        help="Target number of samples")
    args = parser.parse_args()

    middle = []  # correct_num == 1 or 2
    hard = []    # correct_num == 0
    all_samples = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            all_samples.append(x)
            c = x["correct_num"]
            if c == 1 or c == 2:
                middle.append(x)
            elif c == 0:
                hard.append(x)

    print(f"Total scored samples: {len(all_samples)}")
    print(f"  Middle (1/3 or 2/3): {len(middle)}")
    print(f"  Hard   (0/3):       {len(hard)}")

    selected = []
    # 优先取 middle
    selected.extend(middle[:args.target])

    # 若不足则从 hard 中补足
    if len(selected) < args.target:
        need = args.target - len(selected)
        selected.extend(random.sample(hard, min(need, len(hard))))

    random.shuffle(selected)

    # 输出最终选择
    with open(args.output, "w", encoding="utf-8") as f:
        for x in selected:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\nFinal selected: {len(selected)} samples (target: {args.target})")
    if len(selected) > 0:
        final_dist = Counter([x["correct_num"] for x in selected])
        print("Correct_num distribution in final selection:")
        for k in sorted(final_dist.keys()):
            print(f"  {k}/3: {final_dist[k]} samples ({final_dist[k]/len(selected)*100:.1f}%)")

    # 打印前 3 条样本的 correct_num 以检查
    if selected:
        print("\nFirst 3 selected samples (correct_num):")
        for i, item in enumerate(selected[:3]):
            print(f"  Sample {i+1}: correct_num = {item['correct_num']}")

if __name__ == "__main__":
    main()
