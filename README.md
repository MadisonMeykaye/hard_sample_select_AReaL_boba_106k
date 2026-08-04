# AReaL-boba GRPO 数据预处理流水线

本项目提供了一套完整的、工程化的数据预处理流程，用于从 [AReaL-boba-106k](https://huggingface.co/datasets/AReaL-CS/AReaL-boba-106k) 数据集中筛选高质量样本，供 GRPO（Group Relative Policy Optimization）训练使用。  
整个流程模拟了真实 RL 训练链路，适合作为 **大模型 RL / RLHF 方向实习项目** 的展示。

---

## 📌 项目目标

- 从原始 10.6 万条数学推理数据中，通过 **Rollout + 答案匹配** 自动标注每个 prompt 下模型生成的 3 个回答的正确性。
- 基于正确数（`correct_num`）筛选出 **中等难度**（1/3 或 2/3 正确）及部分困难（0/3）样本，用于 GRPO 微调。
- 提供清晰的模块化脚本，支持命令行参数，便于调试、复用和扩展。

---

## 🧩 整体流程

```
AReaL-boba-106k.jsonl
        │
        ▼
  sample_15k.py        # 随机采样 15k 条
        │
        ▼
 sampled_15k.jsonl
        │
        ▼
 rollout_vllm.py      # vLLM 6卡并行生成 3 个回答
        │
        ▼
   rollout.jsonl
        │
        ▼
  check_answer.py     # 提取 \boxed{} 并判对错，计算 correct_num
        │
        ▼
    scored.jsonl
        │
        ▼
   select_5k.py       # 优先取 1/3 或 2/3，不足补 0/3，打乱输出
        │
        ▼
grpo_train_5k.jsonl   # 可直接用于 GRPO 训练
```

---

## 📁 目录结构

```
hard_sample_select/
├── sample_15k.py          # 随机采样
├── rollout_vllm.py        # vLLM 批量生成
├── check_answer.py        # 答案判定（核心）
├── select_5k.py           # 根据正确数筛选
└── output/                # 所有中间及最终输出文件
    ├── sampled_15k.jsonl
    ├── rollout.jsonl
    ├── scored.jsonl
    └── grpo_train_5k.jsonl
```

---

## ⚙️ 环境依赖

- Python 3.10+
- vLLM ≥ 0.6.0
- transformers, torch, tqdm
- 6 张 NVIDIA RTX 3090（24GB）

安装依赖：
```bash
pip install vllm transformers torch tqdm
```

---

## 🚀 快速开始

### 1. 修改默认路径（可选）

每个脚本都支持命令行参数，你可以直接运行默认配置，也可以按需指定：

| 脚本 | 关键参数 |
|------|----------|
| `sample_15k.py` | `--input`, `--output`, `--sample-num`, `--seed` |
| `rollout_vllm.py` | `--model`, `--input`, `--output`, `--tp-size`, `--n`, `--temperature`, `--top-p`, `--max-tokens`, `--gpu-mem-util` |
| `check_answer.py` | `--input`, `--output` |
| `select_5k.py` | `--input`, `--output`, `--target` |

### 2. 依次运行

```bash
# 1. 采样 15k
python sample_15k.py

# 2. 使用 6 卡 vLLM 生成 (请确保 CUDA_VISIBLE_DEVICES 正确)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python rollout_vllm.py

# 3. 打分
python check_answer.py

# 4. 筛选 5k
python select_5k.py
```

执行完毕后，最终训练数据位于 `output/grpo_train_5k.jsonl`。

### 3. 自定义运行（示例）

```bash
# 采样 20k，指定不同输入
python sample_15k.py --input /data/my.jsonl --sample-num 20000 --output ./my_sampled.jsonl

# 使用 4 卡，每个 prompt 生成 5 个回答
python rollout_vllm.py --model /models/Qwen2.5-7B --tp-size 4 --n 5

# 筛选 3k 样本
python select_5k.py --target 3000 --input ./output/scored.jsonl
```

---

## 🔍 核心模块详解

### `check_answer.py` —— 答案匹配逻辑

- 从模型回答中提取 `\boxed{...}` 内容；若没有，则直接使用整个回答。
- 对标准答案同样处理：优先提取 `\boxed{}`，否则直接使用原始 `answer` 字段。
- 执行归一化（去除空格、LaTeX 命令、`$` 符号等）后进行字符串比较。
- 支持通过 `is_equivalent` 函数扩展更复杂的数学等价判断（如分数化简）。

**输出**：每条记录新增 `correct`（长度为 3 的 0/1 列表）和 `correct_num`（0~3 的整数）。

---

### `select_5k.py` —— 难度采样策略

- **中等难度（middle）**：`correct_num == 1` 或 `2`，优先选取。
- **困难样本（hard）**：`correct_num == 0`，作为补充。
- 若中等样本不足目标数，则从困难样本中随机补足。
- 最终打乱顺序，防止数据偏见。

所有脚本均会在运行结束时打印统计信息（样本数量、`correct_num` 分布、前几条样本预览），方便您目视检查。

---

## 📊 输出数据格式

`grpo_train_5k.jsonl` 中每行是一个 JSON 对象，结构如下：

```json
{
  "data": {
    "prompt": "Solve the equation: ...",
    "answer": "\\boxed{42}"
  },
  "generations": [
    "The answer is 42.",
    "I think it's 42.",
    "42"
  ],
  "correct": [1, 1, 1],
  "correct_num": 3
}
```

该格式可直接用于 GRPO 训练脚本，其中 `generations` 可作为模型采样输出，`correct` 可作为奖励信号。

---

## 📝 注意事项

- **显存**：6×3090（24GB）生成 15k 条 * 3 个回答时，`gpu_memory_utilization` 建议设为 0.85~0.9，若 OOM 可降至 0.8 或减小 `max_tokens`。
- **答案格式**：本脚本假设标准答案中包含 `\boxed{}` 或纯文本答案。如果数据格式不同（如仅含 `\frac{...}`），请调整 `extract_box_answer` 或 `is_equivalent` 函数。
- **随机性**：采样和筛选均设置了固定随机种子（默认为 42），保证可复现性。

---

## 🤝 贡献与拓展

本项目可直接作为您简历中 **“大模型强化学习数据工程”** 的示例。  
您还可以在此基础上：
- 添加更多 Reward 信号（如格式正确性、推理步骤连贯性）。
- 集成主流 GRPO 训练框架（如 TRL、OpenRLHF）。
- 进行对比实验（中等难度 vs 困难样本），验证采样策略对模型最终性能的影响。

---

## 📄 License

本项目仅供学习与研究使用，原始数据集版权归 AReaL-CS 所有。

---

**Happy RL Tuning!** 🎉
