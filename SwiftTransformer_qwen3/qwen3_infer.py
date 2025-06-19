import torch
import numpy as np
from torch.classes.gpt_ops import Qwen3Op
from scripts.encode_input import encode_text
from scripts.decode_output import decode_tokens
from scripts.gpt_token_encoder import get_encoder

# === 配置模型超参数（应与 Qwen3-8B 配置一致） ===
num_layers = 32
hidden_size = 4096
num_heads = 32
num_kv_heads = 32
vocab_size = 151936
inter_size = 11008
max_batch_size = 1
max_seq_len = 2048
block_size = 16
inference_dtype = "fp16"

# === 加载 tokenizer ===
encoder = get_encoder()  # 使用 Qwen3 tokenizer 的 bpe 分词器
prompt = "今天天气怎么样？"
input_ids = encode_text(prompt, encoder)  # List[int]
input_ids = torch.tensor([input_ids], dtype=torch.int32, device="cuda")  # (1, seq_len)

# === 初始化 Qwen3 模型 ===
model = Qwen3Op(
    num_layers, hidden_size, num_heads, num_kv_heads,
    vocab_size, inter_size, inference_dtype,
    max_batch_size, max_seq_len, [block_size]
)

model.load_weight("path/to/your/qwen3-8b-swift-weights")  # 👈 修改为你的权重目录

# === 构造缓存 KV cache / block table ===
batch_size, seq_len = input_ids.shape
total_tokens = seq_len
num_blocks = (seq_len + block_size - 1) // block_size

# KV cache 初始化
k_cache = torch.zeros((num_layers, max_batch_size, num_blocks, block_size, hidden_size // num_heads), dtype=torch.float16, device="cuda")
v_cache = torch.zeros_like(k_cache)

# Block table 初始化（索引映射）
block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(batch_size, 1)  # (B, n_blocks)

# First token index（即 seq_len-1）
first_token_indexes = torch.tensor([seq_len - 1], dtype=torch.int32, device="cuda")

# === 模型推理 ===
output = model.forward(input_ids, first_token_indexes, k_cache, v_cache, block_table)  # (1, 1)
output_ids = output[0].tolist()  # 提取单个输出 token id

# === 解码输出 ===
text = decode_tokens(output_ids, encoder)
print("🌟 模型回复:", text)
