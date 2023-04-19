import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import DataParallel
import torch
import numpy as np
import imageio
from tqdm import tqdm
from scipy import stats
from label_dict import protein_to_num_full
from scipy.spatial.distance import cdist
from yaml_tfms import tfms_from_config
import vision_transformer as vits
import yaml
import os
from pathlib import Path
import file_dataset
from sklearn.preprocessing import StandardScaler
from utils import get_dataset
import argparse

protein_to_num = protein_to_num_full
num_proteins = len(protein_to_num)

parser = argparse.ArgumentParser("Get embeddings from model")
parser.add_argument("--config", type=str, default=".", help="path to config file")
parser.add_argument(
    "--batch_size_per_gpu", type=int, default=None, help="path to config file"
)
parser.add_argument(
    "--pretrained_weights",
    type=str,
    default=None,
    help="pretrained weights, if different than config",
)
parser.add_argument(
    "--output_prefix", type=str, default=None, help="path to config file"
)
parser.add_argument("--dataset", type=str, default=None, help="path to config file")

args = parser.parse_args()
config = yaml.safe_load(open(args.config, "r"))
config["embedding"]["HEAD"] = (
    True if "HEAD" in list(config["embedding"].keys()) else False
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = vits.__dict__[config["model"]["arch"]](
    img_size=[224],
    patch_size=config["model"]["patch_size"],
    num_classes=0,
    in_chans=config["model"]["num_channels"],
)
embed_dim = model.embed_dim

# TODO: check in the following line is needed
# model = utils.MultiCropWrapper(model,DINOHead(embed_dim, config['model']['out_dim'], config['model']['use_bn_in_head']))
for p in model.parameters():
    p.requires_grad = False
model.eval()

if type(args.pretrained_weights) is type(None):
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
            teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}
        msg = model.load_state_dict(teacher, strict=False)
    else:
        student = state_dict
        if not config["embedding"]["HEAD"] == True:
            student = {k.replace("module.", ""): v for k, v in student.items()}
            student = {k.replace("backbone.", ""): v for k, v in student.items()}
        student = {k.replace("0.", ""): v for k, v in student.items()}
        msg = model.load_state_dict(student, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model = model.cuda()
    model = model.eval()
    model = DataParallel(model)
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

# model.to(device)
# model = torch.nn.DataParallel(model.cuda())
_, _, _, transform = tfms_from_config(config)

if type(args.dataset) is type(None):
    dataset_path = Path(config["kaggle_test_data"]["df_path"])
else:
    dataset_path = args.dataset

if config["kaggle_test_data"]["averaged_features"]:
    if args.output_prefix is None:
        output_path = config["kaggle_test_data"]["single_cell_kaggle_features_path"]
    else:
        output_path = args.output_prefix
    Path(output_path).parent.absolute().mkdir(exist_ok=True)
else:
    if args.output_prefix is None:
        output_path = config["kaggle_test_data"]["whole_image_kaggle_features_path"]
    else:
        output_path = args.output_prefix
    Path(output_path).parent.absolute().mkdir(exist_ok=True)

reader = file_dataset.pandas_reader_no_labels
loader = file_dataset.image_modes[config["model"]["image_mode"]]
FileList = file_dataset.data_loaders[config["model"]["datatype"]]

dataset = FileList(
    dataset_path,
    root="",
    transform=transform,
    flist_reader=reader,
    loader=loader,
    training=False,
    with_labels=False,
    balance=False,
    single_cells=True,
)

sampler = torch.utils.data.SequentialSampler(dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    sampler=sampler,
    batch_size=config["model"]["batch_size_per_gpu"]
    if type(args.batch_size_per_gpu) is type(None)
    else args.batch_size_per_gpu,
    num_workers=config["kaggle_test_data"]["num_workers"],
    pin_memory=True,
)

IDs = []
impaths = []
all_features = None
all_features = torch.zeros(len(dataset), 768)
i = 0
for images, ID, impath in tqdm(data_loader):
    if isinstance(images, list):
        # Compatibility for crops and multi-views
        with torch.no_grad():
            f = torch.stack([model(i.to(device)) for i in images])
            f = torch.transpose(f, 0, 1)
            features = torch.reshape(f, (f.shape[0], f.shape[1] * f.shape[2]))
    else:
        with torch.no_grad():
            features = model(images.to(device))

    if all_features == None:
        all_features = features.cpu()
    else:
        all_features[i : i + len(features), :] = features.detach().cpu()
    i += len(features)
    IDs.extend(ID)
    impaths.extend(impath)
    if (i % 1000) == 0:
        torch.save((all_features, IDs, impaths), output_path)
        # torch.save((all_features, IDs), output_path)

if config["kaggle_test_data"]["averaged_features"]:
    averaged_features = []
    new_IDs = np.array(pd.DataFrame(IDs)[0].unique())
    all_IDs = np.array(IDs)
    for ID in new_IDs:
        indices = np.where(all_IDs == ID)[0]
        averaged_features.append(all_features[indices, :].mean(axis=0))
    averaged_features = torch.stack(averaged_features)
    IDs = new_IDs
    all_features = averaged_features
    torch.save((all_features, IDs), output_path)

# torch.save((all_features, IDs), output_path)
torch.save((all_features, IDs, impaths), output_path)
print(f"finished calculating features for {output_path}")
