import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Rollout with vLLM")
    parser.add_argument("--model", default="/ssd1/xueqili/models/Qwen2.5-7B-Instruct",
                        help="Path to model")
    parser.add_argument("--input", default="./output/sampled_15k.jsonl",
                        help="Input jsonl file")
    parser.add_argument("--output", default="./output/rollout.jsonl",
                        help="Output jsonl file")
    parser.add_argument("--tp-size", type=int, default=6,
                        help="Tensor parallel size")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of generations per prompt")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens to generate")
    parser.add_argument("--gpu-mem-util", type=float, default=0.9,
                        help="GPU memory utilization")
    args = parser.parse_args()

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} prompts")

    prompts = [x["prompt"] for x in data]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_tokens,
    )

    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, params)

    with open(args.output, "w", encoding="utf-8") as f:
        for item, out in tqdm(zip(data, outputs), total=len(data)):
            generations = [x.text for x in out.outputs]
            record = {"data": item, "generations": generations}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Rollout completed. Saved to {args.output}")

    # 预览第一条生成的样本
    if outputs:
        print("\nPreview first generation:")
        print(f"Prompt: {data[0]['prompt'][:200]}...")
        print(f"Generations: {outputs[0].outputs[0].text[:200]}...")

if __name__ == "__main__":
    main()
