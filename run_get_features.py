import os
import sys
import argparse
import random
import colorsys
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import skimage.io
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.nn import DataParallel
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import yaml
from functools import partial  # (!)
from utils.yaml_tfms import tfms_from_config
import utils.utils
import utils.vision_transformer as vits
from archs import xresnet as cell_models  # (!)
from utils.vision_transformer import DINOHead
from utils import file_dataset

try:
    from get_wair_model import get_wair_model
except:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get embeddings from model")
    parser.add_argument("--config", type=str, default=".", help="path to config file")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="pretrained weights, if different than config",
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None, help="path to config file"
    )
    parser.add_argument("--gpus", type=str, default=".", help="path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="path to config file")
    parser.add_argument("--whole", action="store_true", help="path to config file")

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))

    if args.output_prefix is None:
        output_path = f'{config["embedding"]["output_path"]}'
    else:
        output_path = args.output_prefix
    Path(output_path).parent.absolute().mkdir(exist_ok=True)
    print(f"output_path is {output_path}")

    # TODO: fix these temp compatibility patches:
    if not "HEAD" in list(config["embedding"].keys()):
        print(
            "Please see line 55 in run_get_embeddings.py for additional arguments that can be used to run the full backbone+HEAD model"
        )
    config["embedding"]["HEAD"] = (
        True if "HEAD" in list(config["embedding"].keys()) else False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    wair_model_name_list = [
        "DenseNet121_change_avg_512_all_more_train_add_3_v2",
        "DenseNet121_change_avg_512_all_more_train_add_3_v3",
        "DenseNet121_change_avg_512_all_more_train_add_3_v5",
        "DenseNet169_change_avg_512_all_more_train_add_3_v5",
        "se_resnext50_32x4d_512_all_more_train_add_3_v5",
        "Xception_osmr_512_all_more_train_add_3_v5",
        "ibn_densenet121_osmr_512_all_more_train_add_3_v5_2",
    ]

    if config["model"]["model_type"] == "DINO":
        if config["model"]["arch"] in vits.__dict__.keys():
            # model = vits.__dict__[config['model']['arch']](img_size=[112], patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            # model = vits.__dict__[config['model']['arch']](img_size=[512], patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            model = vits.__dict__[config["model"]["arch"]](
                img_size=[224],
                patch_size=config["model"]["patch_size"],
                num_classes=0,
                in_chans=config["model"]["num_channels"],
            )
            # model = vits.__dict__[config['model']['arch']](img_size=[224], patch_size=config['model']['patch_size'], num_classes=0, in_chans=config['model']['num_channels'])
            embed_dim = model.embed_dim
        elif config["model"]["arch"] in cell_models.__dict__.keys():
            model = partial(
                cell_models.__dict__[config["model"]["arch"]],
                c_in=config["model"]["num_channels"],
            )(False)
            embed_dim = model[-1].in_features
            model[-1] = nn.Identity()

        if config["embedding"]["HEAD"] == True:
            model = utils.MultiCropWrapper(
                model,
                DINOHead(
                    embed_dim,
                    config["model"]["out_dim"],
                    config["model"]["use_bn_in_head"],
                ),
            )
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)
        if args.pretrained_weights is None:
            pretrained_weights = config["embedding"]["pretrained_weights"]
            print(f'loaded {config["embedding"]["pretrained_weights"]}')
        else:
            pretrained_weights = args.pretrained_weights
            print(f"loaded {args.pretrained_weights}")
        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if "teacher" in state_dict:
                teacher = state_dict["teacher"]
                if not config["embedding"]["HEAD"] == True:
                    teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
                    teacher = {
                        k.replace("backbone.", ""): v for k, v in teacher.items()
                    }
                msg = model.load_state_dict(teacher, strict=False)
            else:
                student = state_dict
                if not config["embedding"]["HEAD"] == True:
                    student = {k.replace("module.", ""): v for k, v in student.items()}
                    student = {
                        k.replace("backbone.", ""): v for k, v in student.items()
                    }
                student = {k.replace("0.", ""): v for k, v in student.items()}
                msg = model.load_state_dict(student, strict=False)

            for p in model.parameters():
                p.requires_grad = False
            model = model.cuda()
            model = model.eval()
            model = DataParallel(model)
            # model = DataParallel(model, device_ids=[eval(args.gpus)])
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[eval(args.gpus)])
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    pretrained_weights, msg
                )
            )
        else:
            print(
                "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
            )
            quit()
    elif config["model"]["model_type"] in wair_model_name_list:
        model = get_wair_model(config["model"]["model_type"], fold=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

    _, _, _, transform = tfms_from_config(config)

    if type(args.dataset) is type(None):
        print(f'df_path is {config["embedding"]["df_path"]}')
        dataset_path = Path(config["embedding"]["df_path"])
    else:
        print(f"df_path is {args.dataset}")
        dataset_path = args.dataset

    loader = file_dataset.image_modes[config["model"]["image_mode"]]

    reader = file_dataset.readers[config["embedding"]["embedding_has_labels"]]

    FileList = file_dataset.data_loaders[config["model"]["datatype"]]

    dataset = FileList(
        dataset_path,
        transform=transform,
        flist_reader=reader,
        balance=False,
        loader=loader,
        training=False,
        with_labels=config["embedding"]["embedding_has_labels"],
        root=config["model"]["root"],
        # The target labels are the column names of the protein localizationsm
        # used to create the multilabel target matrix
        target_labels=config['embedding']['target_labels'],
    )

    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config["model"]["batch_size_per_gpu"],
        num_workers=config["embedding"]["num_workers"],
        pin_memory=True,
    )

    labels = None
    all_features = None
    running_index = 0

    # Main feature extraction loop
    for record in tqdm(data_loader):
        # Decode record
        if labels is None:
            labels = [[] for s in record[1:]]
        for ind, label in enumerate(record[1:]):
            labels[ind].extend(record[1 + ind])
        images = record[0]
        # Run model
        if isinstance(images, list):
            # Compatibility for crops and multi-views
            with torch.no_grad():
                f = torch.stack([model(img.to(device)) for img in images])
                f = torch.transpose(f, 0, 1)
                features = torch.reshape(f, (f.shape[0], f.shape[1] * f.shape[2]))
                del f
        else:
            # Single image
            with torch.no_grad():
                features = model(images.to(device))
        # Append features
        if all_features == None:
            all_features = torch.zeros(len(dataset), features.shape[1])
        all_features[running_index : running_index + len(features), :] = features.detach().cpu()
        running_index += len(features)
        del images, record, features

    # Save results
    result = [all_features]
    for l in labels:
        result.append(l)
    torch.save(result, output_path)
