from transformers import PretrainedConfig, OPTForCausalLM, OPTConfig
import torch

torch.set_default_dtype(torch.float16)
config = OPTConfig.from_pretrained("intlsy/opt-175b-hyperparam")
model = OPTForCausalLM(config)
model.save_pretrained("/dev/shm/forged-175b-model")
