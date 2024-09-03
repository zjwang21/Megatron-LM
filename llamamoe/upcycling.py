import argparse
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Receive upcycling model's args")
    parser.add_argument("--model_path", default='meta-llama/Llama-2-7b-hf', type=str, help="original model path")
    parser.add_argument("--output_path", default=None, type=str, help="upcycled model ckpt save path")
    parser.add_argument("--num_experts", default=4, type=int, help="upcycled model num experts")

    # Parse the arguments
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.save_pretrained(args.output_path)
    
    config = MixtralConfig.from_pretrained(args.model_path)
    setattr(config, 'model_type', 'mixtral')
    setattr(config, 'architectures', ["MixtralForCausalLM"])
    setattr(config, 'num_local_experts', args.num_experts)
    config.save_pretrained(args.output_path)
    print(config)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    ckpt = model.state_dict()
    layers = len(model.model.layers)

    output = {k:v for k,v in ckpt.items() if 'mlp' not in k}
    for i in range(layers):
        for k in ckpt:
            if 'mlp' in k and ('layers.' + str(i) + '.') in k:
                for j in range(args.num_experts):
                    if 'mlp.gate_proj' in k:
                        output[k.replace('mlp.gate_proj', 'block_sparse_moe.experts.' + str(j) + '.w1')] = ckpt[k]
                    if 'mlp.up_proj' in k:
                        output[k.replace('mlp.up_proj', 'block_sparse_moe.experts.' + str(j) + '.w3')] = ckpt[k]
                    if 'mlp.down_proj' in k:
                        output[k.replace('mlp.down_proj', 'block_sparse_moe.experts.' + str(j) + '.w2')] = ckpt[k]

    torch.save(output, os.path.join(args.output_path, "pytorch_model.bin"))

if __name__ == "__main__":
    main()

