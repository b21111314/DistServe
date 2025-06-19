"""
This file is used for converting weights from other models (e.g. OPT, LLaMA2)
into SwiftTransformer's format.

Example usage:
- Converting a single, unsharded weight:
  python3 converter.py --input /path/to/weight.pt --output /path/to/output --dtype fp16 --model opt
- Converting a sharded weight:
  python3 converter.py --input /path/to/weight_*.pt --output /path/to/output --dtype fp16 --model llama2

For the detailed workflow, please refer to comments in `converter_lib.py`
"""
import os, sys, argparse, re
from glob import glob
from typing import List, Optional

import torch
import lib.converter_lib as converter_lib

assert __name__ == "__main__"

def load_opt_weight(input: str) -> dict[str, torch.Tensor]:
    files = glob(input)
    if len(files) == 1:
        # unsharded weight. Load it directly
        return torch.load(files[0], torch.device("cpu"))["model"]
    
    def tensorMergeFunc(key: str, tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        dim0_shard_regex = re.compile("embed_tokens|ffn_layernorm|fc1")
        dim1_shard_regex = re.compile("(fc2|out_proj).weight")
        shared_regex = re.compile(
            "embed_positions|layer_norm|(fc2|out_proj).bias|output_projection|version"
        )
        to_ignore_regex = re.compile("decoder.version")
        if to_ignore_regex.search(key):
            # This weight should be ignored
            return None
        elif "qkv_proj.weight" in key:
            hidden_size = tensor_list[0].size(-1)
            return torch.cat(list(map(lambda x: x.view(3, -1, hidden_size), tensor_list)), dim=1).view(-1, hidden_size)
        elif "qkv_proj.bias" in key:
            return torch.cat(list(map(lambda x: x.view(3, -1), tensor_list)), dim = 1).view(-1)
        elif dim0_shard_regex.search(key):
            # This weight is sharded along dim 0
            return torch.cat(tensor_list, dim=0)
        elif dim1_shard_regex.search(key):
            # This weight is sharded along dim 1
            return torch.cat(tensor_list, dim=1)
        elif shared_regex.search(key):
            # This weight is shared across all shards
            return tensor_list[0]
        else:
            raise ValueError(f"Unrecognized weight key: {key}")

    result = converter_lib.reshardWeight(
        files,
        lambda x: x["model"],
        tensorMergeFunc
    )

    return result

def load_llama2_weight(input: str) -> dict[str, torch.Tensor]:
    files = glob(input)
    if len(files) == 1:
        # unsharded weight. Load it directly
        return torch.load(files[0], torch.device("cpu"))["model"]

    def tensorMergeFunc(key: str, tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        dim0_shard_regex = re.compile("\
layers.(\d+).feed_forward.w1.weight|\
layers.(\d+).feed_forward.w3.weight|\
layers.(\d+).attention.w(q|k|v).weight|\
output.weight")
        dim1_shard_regex = re.compile("\
layers.(\d+).feed_forward.w2.weight|\
layers.(\d+).attention.wo.weight|\
tok_embeddings.weight")
        shared_regex = re.compile("\
layers.(\d+).attention_norm.weight|\
layers.(\d+).ffn_norm.weight|\
norm.weight")
        to_ignore_regex = re.compile("rope.freqs")
        if to_ignore_regex.search(key):
            return None
        elif dim0_shard_regex.search(key):
            # This weight is sharded along dim 0
            return torch.cat(tensor_list, dim=0)
        elif dim1_shard_regex.search(key):
            # This weight is sharded along dim 1
            return torch.cat(tensor_list, dim=1)
        elif shared_regex.search(key):
            # This weight is shared across all shards
            return tensor_list[0]
        else:
            raise ValueError(f"Unrecognized weight key: {key}")

    result = converter_lib.reshardWeight(
        files,
        lambda x: x,
        tensorMergeFunc
    )

    return result

def load_qwen3_weight(input: str) -> dict[str, torch.Tensor]:
    files = glob(input)
    if len(files) == 1:
        return torch.load(files[0], torch.device("cpu"))["model"]

    def tensorMergeFunc(key: str, tensor_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        适配 Qwen3-8B 模型的分片规则：
        - qkv_proj 是合并的（shape=[3 * num_heads * head_dim, hidden_dim]）
        - 归一化与嵌入层通常共享
        - Rotary Embedding 可忽略
        """
        to_ignore_regex = re.compile(r"rotary_emb.inv_freq")

        dim0_shard_regex = re.compile(
            r"layers\.(\d+)\.mlp\.(gate_proj|up_proj)\.weight"
        )
        dim1_shard_regex = re.compile(
            r"layers\.(\d+)\.mlp\.down_proj\.weight|"      # FFN down
            r"layers\.(\d+)\.self_attn\.o_proj\.weight|"   # Attention out_proj
            r"tok_embeddings\.weight"
        )
        shared_regex = re.compile(
            r"layers\.(\d+)\.input_layernorm\.weight|"     # Pre-attn norm
            r"layers\.(\d+)\.post_attention_layernorm\.weight|"  # Post-attn norm
            r"norm\.weight"                                # Final norm
        )

        if to_ignore_regex.search(key):
            return None
        elif "qkv_proj.weight" in key:
            # qkv_proj.shape = [3 * num_heads * head_dim, hidden_dim], 分片在 dim=0
            return torch.cat(tensor_list, dim=0)
        elif "qkv_proj.bias" in key:
            # qkv bias.shape = [3 * num_heads * head_dim], 分片在 dim=0
            return torch.cat(tensor_list, dim=0)
        elif dim0_shard_regex.search(key):
            return torch.cat(tensor_list, dim=0)
        elif dim1_shard_regex.search(key):
            return torch.cat(tensor_list, dim=1)
        elif shared_regex.search(key):
            return tensor_list[0]
        else:
            raise ValueError(f"[Qwen3] Unrecognized weight key: {key}")

    result = converter_lib.reshardWeight(
        files,
        lambda x: x["model"],
        tensorMergeFunc
    )

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert weights from other models into SwiftTransformer's format.\
For example usage please refer to comments at the top of this file.")
    parser.add_argument("--input", type=str, required=True, help="Input checkpoint path or glob")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    parser.add_argument("--dtype", type=str, required=True, help="Output dtype")
    parser.add_argument("--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    input = args.input
    output = args.output
    dtype = args.dtype
    os.makedirs(output, exist_ok=True)

    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    assert dtype in torch_dtype, f"Unknown dtype {dtype}, expected one of {torch_dtype.keys()}"
    dtype = torch_dtype[dtype]
    
    supported_models = {"opt", "llama2","qwen3"}
    assert args.model in supported_models, f"Unknown model {args.model}, expected one of {supported_models}"

    print(f"Converting {input} into torch.jit.script format")

    # Load the state dict (tensor_dict)
    # If the whole model is saved in a single file, then load the state dict directly
    # otherwise, load them separately and merge them into a single state dict
    if len(glob(input)) == 0:
        ValueError(f"Input {input} does not match any files")
        print(f"Input {input} does not match any files")
        exit(1)

    if args.model == "opt":
        state_dict = load_opt_weight(input)
    elif args.model == "llama2":
        state_dict = load_llama2_weight(input)
    elif args.model == "qwen3":
        state_dict = load_qwen3_weight(input)
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    print("Resharding and saving weights")
    converter_lib.convertWeight(output, state_dict, dtype, args.model)
    