import argparse
import yaml
import os

parser = argparse.ArgumentParser("Get embeddings from model")
parser.add_argument("--config", type=str, default=".", help="path to config file")
parser.add_argument(
    "--gpus", type=str, default=".", help='Used GPUs, divided by commas (e.g., "1,2,4")'
)
parser.add_argument(
    "--master_port",
    type=str,
    default="29501",
    help='Used GPUs, divided by commas (e.g., "1,2,4")',
)

args = parser.parse_args()
config = yaml.safe_load(open(args.config, "r"))

num_gpus = len(args.gpus.split(","))
command = f'CUDA_VISIBLE_DEVICES={args.gpus} python3 -m torch.distributed.launch --master_port={args.master_port} --nproc_per_node={num_gpus} main_dino.py --arch {config["model"]["arch"]} --output_dir {config["model"]["output_dir"]} --data_path {config["model"]["data_path"]}  --saveckp_freq {config["model"]["saveckp_freq"]} --batch_size_per_gpu {config["model"]["batch_size_per_gpu"]} --num_channels {config["model"]["num_channels"]} --patch_size {config["model"]["patch_size"]} --local_crops_scale {config["model"]["local_crops_scale"]} --epochs {config["model"]["epochs"]} --config {args.config} --center_momentum {config["model"]["center_momentum"]} --lr {config["model"]["lr"]}  {"--sample_single_cells" if(config["model"]["sample_single_cells"] == True) else ""}'
print(command)

os.system(command)
