import oyaml as yaml
import argparse
import sys
from torchvision import datasets, transforms
from torchvision.transforms import *
from utils.augmentations import *


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)
    parser.add_argument("--config", default=".", type=str)
    return parser


def parse_tfms(config, key):
    tfms = config[key]
    augs = []
    for i in tfms:
        f = globals()[i]
        if config[key][i][0] == True:
            print(f"adding {key}: {i}")
            if isinstance(config[key][i][1], dict):
                f = f(**config[key][i][1])
            elif isinstance(config[key][i][1], list):
                f = f(*config[key][i][1])
            else:
                f = f()
            if i in ["ColorJitter", "ColorJitter_for_RGBA"]:
                print("found color jitter")
                f = RandomApply([f], p=0.8)
            augs.append(f)
    return augs


def tfms_from_config(config):
    jitter = parse_tfms(config, "flip_and_color_jitter_transforms")
    norm = parse_tfms(config, "normalization")

    testing_tfms = parse_tfms(config, "testing_transfo")
    testing_tfms = transforms.Compose(testing_tfms)
    glb_tfms_1 = parse_tfms(config, "global_transfo1")
    global_tfms_1 = transforms.Compose(
        glb_tfms_1 + jitter + parse_tfms(config, "global_aug1") + norm
    )
    glb_tfms_2 = parse_tfms(config, "global_transfo2")
    global_tfms_2 = transforms.Compose(
        glb_tfms_2 + jitter + parse_tfms(config, "global_aug2") + norm
    )
    loc_tfms = parse_tfms(config, "local_transfo")
    local_tfms = transforms.Compose(
        loc_tfms + parse_tfms(config, "local_aug") + jitter + norm
    )  # note different order!

    return global_tfms_1, global_tfms_2, local_tfms, testing_tfms


if __name__ == "__main__":
    parser = argparse.ArgumentParser("test", parents=[get_args_parser()])
    args = parser.parse_args()
    dummy_func(args.config)
