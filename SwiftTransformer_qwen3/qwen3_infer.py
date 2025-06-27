import torch
from transformers import GPT2TokenizerFast
import swifttransformer
from swifttransformer.gpt import GptModel
from swifttransformer.model import GptHyperParam, GptParallelismParam, GptPagedAttnParam
from swifttransformer.weight import load_weight_from_bin
import os

# 1. 模型超参（你可根据实际情况调整）
NUM_LAYERS = 32
HIDDEN_SIZE = 4096
NUM_Q_HEADS = 32
NUM_KV_HEADS = 32
HEAD_DIM = 128
FFN_INTER_DIM = 11008
MAX_POSITION_EMBEDDINGS = 32768

# 2. 初始化 tokenizer（基于 GPT2TokenizerFast）
tokenizer = GPT2TokenizerFast(
    vocab_file="model/vocab.json",
    merges_file="model/merges.txt"
)

tokenizer.pad_token = tokenizer.eos_token

# 3. 构造模型超参对象
hyper_param = GptHyperParam(
    vocab_size=len(tokenizer),
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_q_heads=NUM_Q_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    ffn_inter_dim=FFN_INTER_DIM,
    is_pre_layernorm=True,
    is_rotary_posi_embedding=True,
    is_rmsnorm=True,
    is_gated_ffn=True,
    is_attn_qkv_biased=True,
    is_attn_out_biased=True,
)

# 4. 构造模型并加载权重
model = GptModel(
    hyper_param=hyper_param,
    parallel_param=GptParallelismParam(),  # 单卡测试
    attn_param=GptPagedAttnParam()
)

load_weight_from_bin(model, "model/qwen3-bin", dtype=torch.float16)
model.eval()

# 5. 测试文本
prompt = "你好，世界！请介绍一下你自己。"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids

# 6. 生成（最多生成 100 个 token）
with torch.no_grad():
    for _ in range(100):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs[:, -1, :]  # 最后一个 token 的输出
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

# 7. 输出生成结果
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))