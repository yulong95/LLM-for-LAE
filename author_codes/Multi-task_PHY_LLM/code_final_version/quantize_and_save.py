import json
import os
import torch
import torch.nn as nn
from peft import LoftQConfig, LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import GPT2Model,GPT2Tokenizer

class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def quantize_and_save():

    # Download weights and configure LoRA
    model_name_or_path = "/gpt2-ori"
    save_dir = "/"
    bits = 4
    iter = 5
    rank = 16

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model_name = "gpt2" 
    model =  GPT2Model.from_pretrained(model_name_or_path ,torch_dtype=torch.float16)
    target_modules=["c_attn", "c_proj", "c_fc"] 

    # Config of LoftQ
    loftq_config = LoftQConfig(loftq_bits=bits, loftq_iter=iter)

    lora_config = LoraConfig(
        inference_mode=True,
        r=rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        init_lora_weights="loftq",
        loftq_config=loftq_config,
        fan_in_fan_out=True,
    )

    # Obtain LoftQ model
    lora_model = get_peft_model(model, lora_config)
    base_model = lora_model.get_base_model()

    # Save LoftQ model
    model_name = model_name_or_path.split("/")[-1] + f"-{bits}bit" + f"-{rank}rank"
    base_model_dir = os.path.join(save_dir, model_name)
    lora_model_dir = os.path.join(save_dir, model_name, "loftq_init")

    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    print_model(base_model, "base_model")

    # convert safetensor to bin
    tensors = {}
    with safe_open(os.path.join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    # change adapter_config.json
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir  # This can be a local path or Hub model id
        adapter_config['init_lora_weights'] = True  # Don't apply LoftQ when loading again
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    base_dir, lora_dir = quantize_and_save()
