import argparse
import json
from typing import Dict, Any
import copy

TOTAL_PARAM = 6738415616
IMMIDIATE_DIM = 11008
CONTEXT_LEN = 4096
NUM_BLOCK = 32
ATTENTION_WEIGHT_EACH_PARAM = CONTEXT_LEN * CONTEXT_LEN
MLP_WEIGHT_EACH_PARAM = CONTEXT_LEN * IMMIDIATE_DIM

attention = ["q_proj", "k_proj", "v_proj", "o_proj"]
mlp = ["gate_proj", "up_proj", "down_proj"]

parser = argparse.ArgumentParser(description='Translate config file to another format')
parser.add_argument('--input', type=str, help='Input file')
parser.add_argument('--to_type', 
                    type=str,
                    choices=["rank_to_asvd", "asvd_to_rank"], 
                    help='translate to which type: asvd or my')

def init_config() -> Dict[str, Any]:
    tmp = {
            "q_proj": 1.0,
            "k_proj": 1.0,
            "v_proj": 1.0,
            "o_proj": 1.0,
            "gate_proj": 1.0,
            "up_proj": 1.0,
            "down_proj": 1.0
        }
    config = {
        "blocks": []
    }
    
    for i in range(32):
        config["blocks"].append(copy.deepcopy(tmp))
    
    return config


def translate_to_asvd(my_config):
    asvd_config = {}
    asvd_config["lm_head"] = 1
    
    for idx in range(31, -1, -1):
        block = my_config["blocks"][idx]
        asvd_config[f"model.layers.{idx}.self_attn.q_proj"] = block["q_proj"]
        asvd_config[f"model.layers.{idx}.self_attn.k_proj"] = block["k_proj"]
        asvd_config[f"model.layers.{idx}.self_attn.v_proj"] = block["v_proj"]
        asvd_config[f"model.layers.{idx}.self_attn.o_proj"] = block["o_proj"]
        asvd_config[f"model.layers.{idx}.mlp.gate_proj"] = block["gate_proj"]
        asvd_config[f"model.layers.{idx}.mlp.up_proj"] = block["up_proj"]
        asvd_config[f"model.layers.{idx}.mlp.down_proj"] = block["down_proj"]
    return asvd_config

def translate_to_my(asvd_config):
    config = init_config()
    config['value'] = "rank"
    
    for name, param_ratio in asvd_config.items():
        if name == "lm_head":
            continue
        splits = name.split(".")
        proj = splits[4]
        block = int(splits[2])
        
        if param_ratio == 1:
            rank = CONTEXT_LEN  # Full rank
        else:
            if proj in attention:
                compressed_params = int(ATTENTION_WEIGHT_EACH_PARAM * param_ratio)
                rank = compressed_params // (CONTEXT_LEN + CONTEXT_LEN)
            elif proj in mlp:
                compressed_params = int(MLP_WEIGHT_EACH_PARAM * param_ratio)
                rank = compressed_params // (IMMIDIATE_DIM + CONTEXT_LEN)
        
        if proj == "q_proj":
            config["blocks"][block]["q_proj"] = rank
        elif proj == "k_proj":
            config["blocks"][block]["k_proj"] = rank
        elif proj == "v_proj":
            config["blocks"][block]["v_proj"] = rank
        elif proj == "o_proj":
            config["blocks"][block]["o_proj"] = rank
        elif proj == "gate_proj":
            config["blocks"][block]["gate_proj"] = rank
        elif proj == "up_proj":
            config["blocks"][block]["up_proj"] = rank
        elif proj == "down_proj":
            config["blocks"][block]["down_proj"] = rank
    
    return config


if __name__ == '__main__':
    args = parser.parse_args()
    config_name = args.input.split(".")[0]
    
    if args.to_type == "rank_to_asvd":
        with open(args.input, "r") as f:
            config = json.load(f)
        print("Config {args.input} is loaded.")
        
        asvd_config = translate_to_asvd(config)
        
        # Save to file
        with open(config_name + "_asvd.json", "w") as f:
            json.dump(asvd_config, f)
            print(f"Config is saved as {config_name}_asvd.json")
    elif args.to_type == "asvd_to_rank":
        with open(args.input, "r") as f:
            config = json.load(f)
        print(f"Config {args.input} is loaded.")
        
        my_config = translate_to_my(config)
        
        # Save to file
        with open(config_name + "_rank.json", "w") as f:
            json.dump(my_config, f)
            print(f"Config is saved as {config_name}_rank.json")
            
    else:
        raise ValueError("Unknown type")