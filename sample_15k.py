import json
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Random sample from AReaL-boba-106k")
    parser.add_argument("--input", default="/home/xueqili/project/LLaMA-Factory/data/AReaL-boba-106k.jsonl",
                        help="Input jsonl file")
    parser.add_argument("--output", default="./output/sampled_15k.jsonl",
                        help="Output jsonl file")
    parser.add_argument("--sample-num", type=int, default=15000,
                        help="Number of samples to draw")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Total records: {len(data)}")

    sampled = random.sample(data, min(args.sample_num, len(data)))

    with open(args.output, "w", encoding="utf-8") as f:
        for x in sampled:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Saved {len(sampled)} records to {args.output}")

    # 打印一条样本预览
    if sampled:
        print("\nSample preview (first record):")
        print(json.dumps(sampled[0], ensure_ascii=False, indent=2)[:500] + "...")

if __name__ == "__main__":
    main()
