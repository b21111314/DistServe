import argparse
from distserve import OfflineLLM, SamplingParams
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to Qwen3 model directory', default='model/qwen3-bin')
parser.add_argument('--tokenizer', type=str, help='Path to tokenizer vocab.json', default='model/vocab.json')
args = parser.parse_args()

# 示例提示词
prompts = [
    "你好，请简单介绍一下你自己。",
    "人工智能对社会有什么影响？"
]

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=64,
    stop=["<|endoftext|>", "<|im_end|>"]
)

# 创建 OfflineLLM 实例
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=args.tokenizer,  # vocab.json 路径
    ),
    disagg_parallel_config=DisaggParallelConfig(
        context=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        ),
        decoding=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    ),
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=64,#这里根据GPU内存自行设置
        gpu_memory_utilization=0.3,#这个也是
        cpu_swap_space=1.0
    ),
    context_sched_config=ContextStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    )
)
# 执行推理
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# 打印生成结果
for prompt, step_outputs in zip(prompts, outputs):
    generated_text = ''.join([step.new_token for step in step_outputs])
    print(f"Prompt: {prompt}\nGenerated: {generated_text}\n{'-'*60}")