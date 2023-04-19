# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import numbers
from typing import Tuple, List, Optional
from sklearn.metrics import f1_score
import argparse
import torchvision
import math
from torch import Tensor
import warnings
from collections.abc import Sequence
from PIL import Image
import numpy as np
from torch.nn.functional import sigmoid
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.filters import threshold_otsu
from PIL import Image, ImageFilter, ImageOps
import cuml
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CyclicLR, ConstantLR
import yaml
from matplotlib import cm
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from seaborn import clustermap
from scipy.spatial.distance import squareform
import cuml
from sklearn.metrics import average_precision_score
import pandas as pd
from skimage import io
from label_dict import protein_to_num_single_cells, protein_to_num_full
from label_dict import (
    hierarchical_organization_single_cell_low_level,
    hierarchical_organization_single_cell_high_level,
    hierarchical_organization_whole_image_low_level,
    hierarchical_organization_whole_image_high_level,
)
import matplotlib.ticker as mtick
import label_dict
from sklearn.cross_decomposition import CCA
from sklearn import datasets
from sklearn import decomposition

# import kornia as K
import kornia.geometry.transform.imgwarp as K
import os
import sys
import time
import datetime
import random
import subprocess
from collections import defaultdict, deque
import torch.distributed as dist
import matplotlib.pyplot as plt
from file_dataset import AutoBalancedPrecomputedFeatures
from random import choices
from sklearn.cluster import DBSCAN
import matplotlib.patches as patches
import mantel
import harmonypy as hm

cmap = cm.nipy_spectral


def load_pretrained_weights(
    model, pretrained_weights, checkpoint_key, model_name, patch_size
):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                pretrained_weights, msg
            )
        )
    else:
        print(
            "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
        )
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=True)
        else:
            print(
                "There is no reference weights available for this model => We use random weights."
            )


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g["weight_decay"])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        ## compatibility with xresnet ##
        try:
            backbone[-1], backbone.head = nn.Identity(), nn.Identity()
            print("Caught missing fc of xresnet in MultiCropWrapper")
        except:
            backbone.fc, backbone.head = (
                nn.Identity(),
                nn.Identity(),
            )  # original single line

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx:end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_params_groups(*models):
    regularized = []
    not_regularized = []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


# ====================== Augmentations =========================================

Normalize = transforms.Normalize


class ToTensor:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        else:
            return transforms.ToTensor()(x)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class GaussianBlur_pytorch(object):
    """
    Apply Gaussian Blur to the PIL or Tensor image. The number of channels can be arbitrary.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0, kernel_size=3):
        self.prob = p
        self.blur = transforms.GaussianBlur(kernel_size, sigma=(radius_min, radius_max))

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return self.blur(img)


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


# ====================== PIL-independent transforms ==============================


class Warp_cell(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.ToTensor()(img)
        img = img.unsqueeze(0)
        points_src = torch.Tensor(
            [
                [
                    [0.0, 0.0],
                    [float(img.shape[2]), 0.0],
                    [float(img.shape[2]), float(img.shape[3])],
                    [0, float(img.shape[3])],
                ]
            ]
        )

        shift = max(int(min(img.shape[2], img.shape[3]) / 4), 1)
        points_dst = (points_src).clone()
        corner_ind_0 = np.random.randint(4)
        for corner_ind_1 in [0, 1]:
            val = points_dst[0, corner_ind_0, corner_ind_1]
            if val == 0:
                points_dst[0, corner_ind_0, corner_ind_1] = points_dst[
                    0, corner_ind_0, corner_ind_1
                ] + np.random.randint(0, shift)
            else:
                points_dst[0, corner_ind_0, corner_ind_1] = points_dst[
                    0, corner_ind_0, corner_ind_1
                ] - np.random.randint(0, shift)

        M = K.get_perspective_transform(points_src, points_dst)
        img_warp = K.warp_perspective(
            img.float(), M, dsize=(img.shape[2], img.shape[3]), align_corners=True
        )
        res = img_warp[0, ...]
        if torch.isnan(res).any():
            print(
                "Warning: converting nan to zeros in Warp_cell(). This is a temporary fix for testing."
            )
            res = torch.nan_to_num(res)
        return res


class RandomResizedCenterCrop(torch.nn.Module):
    """
    Crop an image to a given size. Coordinates for crop are randomly drawn from a guassian
    distribution and will hence primarily target the center of the image.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        interpolation (torchvision.transforms.InterpolationMode): Desired interpolation.
        depth (int or float): sampling depth for the generated distribution.
        s (float): desired standard deviation for the generated guassian distribution. Use this to control the degree to which
            crops are biased to originate from the center of the image (lower s -> stronger bias).
    """

    def __init__(
        self,
        size,
        scale,
        interpolation=transforms.InterpolationMode("bilinear"),
        depth=1e6,
        s=1.0,
    ):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )
        self.scale = scale
        self.dist = self.getdistrib(depth, s)
        self.interpolation = interpolation

    def getdistrib(self, depth, s):
        d = torch.randn(int(depth))
        mx = torch.max(d)
        mn = torch.abs(torch.min(d))
        d = d * s
        d = (d + mn) / (mn + mx)
        return d

    @staticmethod
    def get_params(img, scale, dist):
        width, height = img.shape[1:]
        assert (
            width == height
        ), "RandomResizedCenterCrop for now requires square images. Sorry!"
        sz = width
        target_area = sz * torch.empty(1).uniform_(scale[0], scale[1]).item()
        tsz = int(round(target_area))

        draw = torch.randint(0, dist.size()[0], size=(2,))
        i = int(torch.round(dist[draw[0]] * (sz - tsz)))
        j = int(torch.round(dist[draw[1]] * (sz - tsz)))
        return i, j, tsz, tsz

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized. Expected in [...,h,w]
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.dist)
        return transforms.functional.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )


class RandomResizedCrop(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=transforms.InterpolationMode("bilinear"),
    ):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        #         width, height = torchvision.transforms.functional._get_image_size(img)
        width, height = img.shape[1:]
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """

        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transforms.functional.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )


class rescale_protein(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        if np.random.rand() <= self.p:
            random_factor = (np.random.rand() * 2) / img.max()  # scaling
            img[1] = img[1] * random_factor
            return img
        else:
            return img


class Change_contrast(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[ind] = torchvision.transforms.functional.adjust_contrast(
                img[ind][None, ...], factor
            )
        return img


class Single_cell_Resize(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.cell_image_size = size
        self.n = 0

    def forward(self, img):
        self.n = self.n + 1
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        if sum(np.array(img.shape) < 5) > 1:
            print(f"Warning! Very small image passed. Dimensions: {img.shape}")
        # Below deactivated. We now have a strict assumption that img has dimensions [c,w,h]
        # if np.argmin(img.shape) != 0: img = img.permute(2,1,0)
        FA_img = FA_resize(self.cell_image_size)(img)
        return FA_img


class remove_channel(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        img_size = np.array(img).shape
        if min(img_size) < 4:
            return img
        if np.random.rand() <= self.p:
            channel_to_blacken = np.random.choice(
                np.array([0, 2, 3]), 1, replace=False
            )[0]
            img[channel_to_blacken] = torch.zeros(1, *img.shape[1:])
            return img
        else:
            return img


class Change_brightness(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if img.max() == 0:
            return img
        n_channels = img.shape[0]
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img[ind] = torchvision.transforms.functional.adjust_brightness(
                img[ind], factor
            )
        return img


# ====================== FastAI-adapted transforms ==============================


def dihedral(x: torch.Tensor, k):
    if k in [1, 3, 4, 7]:
        x = x.flip(-1)
    if k in [2, 4, 5, 7]:
        x = x.flip(-2)
    if k in [3, 5, 6, 7]:
        x = x.transpose(-1, -2)
    return x


class rnd_dihedral(object):
    def __call__(self, x):
        k = np.random.randint(0, 7)
        return dihedral(x, k)


class Rotate_dihedral(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        x = transforms.ToTensor()(img)
        k = np.random.randint(0, 7)
        x = (dihedral(x, k).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        x = Image.fromarray(np.array(x))
        return x


class self_normalize_without_black_pixels(object):
    def __call__(self, x):
        nan_x = np.where(x > 0, x, np.nan)
        m = torch.Tensor(np.nanmean(nan_x, (-2, -1))).unsqueeze(1).unsqueeze(1)
        s = torch.Tensor(np.nanstd(nan_x, (-2, -1))).unsqueeze(1).unsqueeze(1)
        m = torch.where(torch.isnan(m), 0, m)
        s = torch.where(torch.isnan(s), 0, 1)
        x -= m
        x /= s + 1e-7
        return x


class self_normalize(object):
    def __call__(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x


class to_np(object):
    def __call__(self, x):
        return np.array(x)


def FA2_grid_sample(
    x, coords, mode="bilinear", padding_mode="reflection", align_corners=None
):
    "Resample pixels in `coords` from `x` by `mode`, with `padding_mode` in ('reflection','border','zeros')."
    # coords = coords.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1) # optimize layout for grid_sample
    if mode == "bilinear":  # hack to get smoother downwards resampling
        mn, mx = coords.min(), coords.max()
        # max amount we're affine zooming by (>1 means zooming in)
        z = 1 / (mx - mn).item() * 2
        # amount we're resizing by, with 100% extra margin
        d = min(x.shape[-2] / coords.shape[-2], x.shape[-1] / coords.shape[-1]) / 2
        # If we're resizing up by >200%, and we're zooming less than that, interpolate first
        if d > 1 and d > z:
            # Pytorch > v1.4.x needs an extra argument when calling nn.functional.interpolate to preserve previous behaviour
            if int(torch.__version__[0:4].replace(".", "")) > 14:
                x = F.interpolate(
                    x, scale_factor=1 / d, mode="area", recompute_scale_factor=True
                )
            else:
                x = F.interpolate(x, scale_factor=1 / d, mode="area")
    return F.grid_sample(
        x, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )


def _init_mat(x):
    mat = torch.eye(3, device=x.device).float()
    return mat.unsqueeze(0).expand(x.size(0), 3, 3).contiguous()


def FA2affine_grid(theta, size, align_corners=None):
    #     return TensorFlowField(F.affine_grid(theta, size, align_corners=align_corners))
    return torch.Tensor(F.affine_grid(theta, size, align_corners=align_corners))


def FA2affine_coord(
    x,
    mat=None,
    coord_tfm=None,
    sz=None,
    mode="bilinear",
    pad_mode="reflection",
    align_corners=True,
):
    if mat is None and coord_tfm is None and sz is None:
        return x
    size = (
        tuple(x.shape[-2:])
        if sz is None
        else (sz, sz)
        if isinstance(sz, int)
        else tuple(sz)
    )
    if mat is None:
        mat = _init_mat(x)[:, :2]
    coords = FA2affine_grid(mat, x.shape[:2] + size, align_corners=align_corners)
    #     if coord_tfm is not None: coords = coord_tfm(coords)
    #     return TensorImage(FA2_grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners))
    return torch.Tensor(
        FA2_grid_sample(
            x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners
        )
    )


class FA_resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        im = FA2affine_coord(x[None, ...], sz=self.size)
        return im[0]


# =============================== Other transforms ===============================


def resize_for_5_channels(image, size):
    return Image.merge("RGBA", [c.resize(size) for c in image.split()])


def resize_RGBA(image, size):
    return Image.merge("RGBA", [c.resize(size) for c in image.split()])


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Permute(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img.permute(2, 0, 1)


def resized_crop_for_5_channels(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: int = Image.BILINEAR,
) -> Tensor:

    img = torchvision.transforms.functional.resized_crop(
        img.permute(2, 0, 1),
        top=top,
        left=left,
        height=height,
        width=width,
        size=size,
        align_corners=False,
    ).permute(1, 2, 0)
    return img


def resized_crop(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: List[int],
    interpolation: int = Image.BILINEAR,
) -> Tensor:
    img = img.crop((top, left, top + height, left + width))
    img = resize_RGBA(img, size)
    return img


class RandomResizedCrop_for_5_channels(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=transforms.InterpolationMode("bilinear"),
    ):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        width, height = torchvision.transforms.functional._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return resized_crop_for_5_channels(
            img, i, j, h, w, self.size, self.interpolation
        )


def solarize_for_RGBA(image, threshold=128):
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(i)
        else:
            lut.append(255 - i)
    lut = lut + lut + lut + lut
    return image.point(lut)


class Solarization_for_5_channels(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            partial_image = Image.fromarray(img[:, :, :4].numpy().astype(np.uint8))
            partial_image = solarize_for_RGBA(partial_image)
            return torch.Tensor(
                np.concatenate(
                    (np.array(partial_image), img[:, :, [4]].numpy()), axis=2
                )
            )
        else:
            return img


class Solarization_for_RGBA(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            img = solarize_for_RGBA(img)
            return img
        else:
            return img


class GaussianBlur_for_5_channels(torch.nn.Module):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.random() <= self.prob
        if not do_it:
            return img
        partial_image = Image.fromarray(img[:, :, :4].numpy().astype(np.uint8))

        partial_image = partial_image.filter(
            ImageFilter.GaussianBlur(
                radius=np.random.uniform(self.radius_min, self.radius_max)
            )
        )
        return torch.Tensor(
            np.concatenate((np.array(partial_image), img[:, :, [4]].numpy()), axis=2)
        )


class RandomGrayscale_for_5_channels(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            return torch.Tensor(
                np.concatenate(
                    (
                        np.repeat((img[:, :, :4]).mean(axis=2)[:, :, np.newaxis], 4, 2)
                        .numpy()
                        .astype(np.uint8),
                        img[:, :, [4]],
                    ),
                    axis=2,
                )
            )
        return img


class Single_cell_random_resize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        row_cell_image_size = np.random.randint(180, 224)
        col_cell_image_size = np.random.randint(180, 224)
        img = Image.merge(
            "RGBA",
            [c.resize((row_cell_image_size, col_cell_image_size)) for c in img.split()],
        )
        cell_image_size = 224
        img_shape = (row_cell_image_size, col_cell_image_size)
        upper_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
        lower_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
        left_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
        right_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
        upper_pad += 224 - (upper_pad + lower_pad + img_shape[1])
        left_pad += 224 - (right_pad + left_pad + img_shape[0])
        new_img = np.pad(
            img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        ).astype(np.uint8)
        return Image.fromarray(new_img)


class RandomGrayscale_for_RGBA(torch.nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1) < self.p:
            return Image.fromarray(
                np.repeat(np.array(img).mean(axis=2)[:, :, np.newaxis], 4, 2).astype(
                    np.uint8
                )
            )
        return img


class Threshold_protein(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        if np.random.rand() < self.p:
            channels = list(img.split())
            protein_channel = 1
            arr = np.array(channels[protein_channel])
            # thresh = threshold_otsu(arr)
            thresh = np.random.randint(0, 50)
            channels[protein_channel] = Image.fromarray(np.where(arr > thresh, arr, 0))
            return Image.merge("RGBA", channels)
        return img


class Jigsaw(torch.nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, img):
        img_array = np.array(img)
        if np.random.rand() < self.p:
            rows = np.linspace(0, img_array.shape[0], 5).astype(int)
            cols = np.linspace(0, img_array.shape[0], 5).astype(int)
            all_tiles = []
            for r, _ in enumerate(rows[:-1]):
                for c, _ in enumerate(cols[:-1]):
                    all_tiles.append(
                        img_array[rows[r] : rows[r + 1], cols[c] : cols[c + 1], :]
                    )
            np.random.shuffle(all_tiles)
            new_img_array = np.zeros(img_array.shape)
            for r, _ in enumerate(rows[:-1]):
                for c, _ in enumerate(cols[:-1]):
                    new_img_array[
                        rows[r] : rows[r + 1], cols[c] : cols[c + 1], :
                    ] = all_tiles[r + c * 4]
            new_img = Image.fromarray(new_img_array.astype(np.uint8))
            return new_img
        else:
            return img


class Single_cell_centered(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.cell_image_size = size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        cell_image_size = self.cell_image_size
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.ToTensor()(img)
        new_img = img
        img_shape = new_img.permute(1, 2, 0).shape

        if (img_shape[0] >= img_shape[1]) and (img_shape[0] > cell_image_size):
            new_img = transforms.functional.resized_crop(
                new_img,
                0,
                0,
                img_shape[0],
                img_shape[1],
                (int(img_shape[1] * cell_image_size / img_shape[0]), cell_image_size),
                transforms.InterpolationMode("bilinear"),
            )
        elif (img_shape[1] >= img_shape[0]) and (img_shape[1] > cell_image_size):
            new_img = transforms.functional.resized_crop(
                new_img,
                0,
                0,
                img_shape[0],
                img_shape[1],
                (cell_image_size, (int(img_shape[0] * cell_image_size / img_shape[1]))),
                transforms.InterpolationMode("bilinear"),
            )
        img_shape = new_img.permute(1, 2, 0).shape
        pad_border = 0
        upper_pad = int(
            min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border)
        )
        lower_pad = int(
            min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border)
        )
        left_pad = int(
            min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border)
        )
        right_pad = int(
            min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border)
        )
        new_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        ).transpose(1, 2, 0)
        img_shape = new_img.shape
        upper_pad = int((cell_image_size - img_shape[0]) / 2)
        lower_pad = max(
            cell_image_size - img_shape[0] - upper_pad,
            int((cell_image_size - img_shape[0]) / 2),
        )
        left_pad = int((cell_image_size - img_shape[1]) / 2)
        right_pad = max(
            cell_image_size - img_shape[1] - left_pad,
            int((cell_image_size - img_shape[1]) / 2),
        )
        padded_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        )
        img = transforms.ToTensor()(padded_img)
        return img


class Single_cell_Mirror(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.cell_image_size = size

    def forward(self, img):
        cell_image_size = self.cell_image_size
        new_img = np.array(img)
        img_shape = new_img.shape
        if (img_shape[0] >= img_shape[1]) and (img_shape[0] > cell_image_size):
            new_img = np.array(
                Image.fromarray(new_img).resize(
                    (
                        int(img_shape[1] * cell_image_size / img_shape[0]),
                        cell_image_size,
                    )
                )
            )
        elif (img_shape[1] >= img_shape[0]) and (img_shape[1] > cell_image_size):
            new_img = np.array(
                Image.fromarray(new_img).resize(
                    (
                        cell_image_size,
                        int(img_shape[0] * cell_image_size / img_shape[1]),
                    )
                )
            )
        img_shape = new_img.shape
        pad_border = 10
        upper_pad = int(
            min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border)
        )
        lower_pad = int(
            min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border)
        )
        left_pad = int(
            min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border)
        )
        right_pad = int(
            min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border)
        )
        new_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        ).astype(np.uint8)
        img_shape = new_img.shape
        upper_pad = int((cell_image_size - img_shape[0]) / 2)
        lower_pad = max(
            cell_image_size - img_shape[0] - upper_pad,
            int((cell_image_size - img_shape[0]) / 2),
        )
        left_pad = int((cell_image_size - img_shape[1]) / 2)
        right_pad = max(
            cell_image_size - img_shape[1] - left_pad,
            int((cell_image_size - img_shape[1]) / 2),
        )
        padded_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "reflect"
        ).astype(np.uint8)
        pil_img = Image.fromarray(padded_img)
        return pil_img


class Single_cell_Resize_keep_aspect_ratio(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.cell_image_size = size

    def forward(self, img):
        cell_image_size = self.cell_image_size
        new_img = np.array(img)
        ratio = new_img.shape[0] / new_img.shape[1]
        if ratio > 1:
            new_x_shape = min(int(cell_image_size / ratio), cell_image_size)
            new_y_shape = int(cell_image_size)
        else:
            new_x_shape = int(cell_image_size)
            new_y_shape = min(int(cell_image_size * ratio), cell_image_size)
        new_img = np.array(
            Image.merge(
                "RGBA",
                [
                    c.resize((new_x_shape, new_y_shape))
                    for c in Image.fromarray(new_img).split()
                ],
            )
        )

        img_shape = new_img.shape
        upper_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
        lower_pad = int((max((cell_image_size - (img_shape[0])) / 2, 0)))
        left_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
        right_pad = int((max((cell_image_size - (img_shape[1])) / 2, 0)))
        upper_pad += cell_image_size - (upper_pad + lower_pad + img_shape[0])
        left_pad += cell_image_size - (left_pad + right_pad + img_shape[1])

        new_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        ).astype(np.uint8)
        new_img = new_img[:224, :224, :]
        new_img = Image.fromarray(new_img)
        return new_img
        # return Image.merge('RGBA', [c.resize((self.cell_image_size, self.cell_image_size)) for c in img.split()])


class Rotate_single_cell(torch.nn.Module):
    def __init__(self, p=0.8):
        super().__init__()
        self.p = p

    def forward(self, img):
        if np.random.rand() < (self.p):
            angle = np.random.randint(0, 360)
            img = img.rotate(angle)
        return img


class ColorJitter_for_5_channels(torch.nn.Module):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        super().__init__()
        self.trans = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def forward(self, tensor):
        img = Image.fromarray(tensor[:, :, :4].numpy().astype(np.uint8))
        channel_indices = np.random.choice(range(4), 3, replace=False)
        channels = list(img.split())
        jittered_image = Image.merge("RGB", [channels[i] for i in channel_indices])
        jittered_image = self.trans(jittered_image)
        jittered_channels = jittered_image.split()
        for ind, channel_ind in enumerate(channel_indices):
            channels[channel_ind] = jittered_channels[ind]
        for ind, c in enumerate(channels):
            channels[ind] = np.array(channels[ind])[:, :, np.newaxis]
        return torch.Tensor(
            np.concatenate(
                (
                    np.concatenate(channels, axis=2),
                    tensor[:, :, 4].numpy()[:, :, np.newaxis],
                ),
                axis=2,
            )
        )


class ColorJitter_for_RGBA(torch.nn.Module):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        super().__init__()
        self.trans = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def forward(self, img):
        channel_indices = np.random.choice(range(4), 3, replace=False)
        channels = list(img.split())
        jittered_image = Image.merge("RGB", [channels[i] for i in channel_indices])
        jittered_image = self.trans(jittered_image)
        jittered_channels = jittered_image.split()
        for ind, channel_ind in enumerate(channel_indices):
            channels[channel_ind] = jittered_channels[ind]
        return Image.merge("RGBA", channels)


class Get_specific_channel(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, img):
        result = img[[self.c], :, :]
        return result


# ============================== Classification =======================================
class MLLR:
    def __init__(self, max_iter=100, num_classes=None):
        self.index = None
        self.y = None
        self.num_classes = num_classes
        self.max_iter = max_iter

    def fit(self, X, y):
        if type(self.num_classes) == type(None):
            self.num_classes = np.array(y).shape[1]

        self.classifiers = [
            LogisticRegression(max_iter=self.max_iter) for c in range(self.num_classes)
        ]

        pbar = tqdm(enumerate(self.classifiers))
        for ind, c in pbar:
            pbar.set_description(f"training LinearRegressor for class {ind}")
            if y[:, ind].mean() in [0, 1]:
                print(f"No two classes for ind {ind}!")
            else:
                c.fit(X, y[:, ind])

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.num_classes))
        for ind, c in enumerate(self.classifiers):
            try:
                predictions[:, ind] = c.predict(X)
            except:
                pass
        return predictions


def threshold_output(prediction, threshold=0.5, use_sigmoid=False):
    if use_sigmoid:
        prediction = sigmoid(prediction)
    return prediction > threshold


class Multilabel_classifier(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_units=500,
        with_sigmoid=True,
        num_blocks=3,
    ):
        super().__init__()
        num_blocks = num_blocks
        layers = []
        layers.extend(
            [
                nn.Linear(num_features, hidden_units),
                nn.BatchNorm1d(hidden_units),
                nn.ReLU(),
                # nn.Dropout(),
            ]
        )
        for i in range(num_blocks):
            layers.extend(
                [
                    nn.Linear(hidden_units, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.ReLU(),
                    nn.Dropout() if i == (num_blocks - 1) else nn.Identity(),
                ]
            )
        layers.append(nn.Linear(hidden_units, num_classes))
        self.layers = torch.nn.Sequential(*layers)
        if with_sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

    def forward(self, x):
        x = self.layers(x)
        x = self.sigmoid(x)
        return x


class residual_clf(nn.Module):
    def __init__(self, n_features, n_classes, n_layers=2, norm_layer=None):
        super().__init__()
        self.layers = torch.nn.Sequential(
            *[
                nn.Linear(
                    # Adding residual input sizes + last layer output size
                    sum([n_features * (max(j, 1)) for j in range(i)]),
                    # Outputing expanding features
                    n_features * (i),
                )
                for i in range(1, n_layers + 1)
            ]
        )
        if norm_layer is not None:
            if norm_layer == "layer":
                norm = torch.nn.LayerNorm
            elif norm_layer == "instance":
                norm = torch.nn.InstanceNorm1d
            self.norm_layers = torch.nn.Sequential(
                *[norm(n_features * (max(i, 1))) for i in range(0, n_layers + 1)]
            )
        else:
            self.norm_layers = []
        self.final = nn.Linear(
            sum([n_features * (max(j, 1)) for j in range(n_layers + 1)]), n_classes
        )

    def forward(self, x):
        Xs = [x]
        if len(self.norm_layers) > 0:
            Xs[0] = self.norm_layers[0](Xs[0])
        for ind, l in enumerate(self.layers):
            inputs = torch.concat(Xs, axis=len(x.shape) - 1)
            res = torch.nn.functional.relu(l(inputs))
            if len(self.norm_layers) > 0:
                res = self.norm_layers[ind + 1](res)
            Xs.append(res)
        return self.final(torch.concat(Xs, axis=len(x.shape) - 1))


class ResBlock(nn.Module):
    def __init__(self, n_units=1024, norm=torch.nn.Identity, skip=True):
        super().__init__()
        self.fc1 = nn.Linear(n_units, n_units)
        self.norm1 = norm(n_units)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(n_units, n_units)
        self.norm2 = norm(n_units)
        self.skip = skip

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.norm2(out)
        if self.skip:
            out += identity
        out = self.relu(out)
        return out


class residual_add_clf(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        n_layers=2,
        n_units=1024,
        norm_layer=None,
        skip=True,
    ):
        super().__init__()
        if norm_layer == "layer":
            self.norm = torch.nn.LayerNorm
            self.norm_layer = self.norm(n_units)
        elif norm_layer == "instance":
            self.norm = torch.nn.InstanceNorm1d
            self.norm_layer = self.norm(n_units)
        elif norm_layer == "batch_norm":
            self.norm = torch.nn.BatchNorm1d
            self.norm_layer = self.norm(1)
        else:
            self.norm = torch.nn.Identity
            self.norm_layer = self.norm(n_units)
        self.first_layer = nn.Linear(n_features, n_units)
        self.layers = torch.nn.Sequential(
            *[
                ResBlock(n_units, self.norm, skip)
                for i in range(int(max(0, (n_layers - 2)) / 2))
            ]
        )
        self.final_layer = nn.Linear(n_units, n_classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        f = self.norm_layer(self.first_layer(x))
        x = torch.nn.functional.relu(f)
        if len(self.layers) > 0:
            x = self.layers(x)
        x = self.final_layer(x)
        return x


class expanding_clf(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.l1 = nn.Linear(n_features, n_features * 2)
        self.l2 = nn.Linear((n_features * 2) + n_features, n_features * 4)
        self.l3 = nn.Linear((n_features * 4) + n_features + (n_features * 2), n_classes)

    def forward(self, x):
        x1 = torch.nn.functional.relu(self.l1(x))
        x2 = torch.nn.functional.relu(self.l2(torch.concat((x1, x), axis=1)))
        x3 = self.l3(torch.concat((x2, x1, x), axis=1))
        return x


class prototyping_clf(nn.Module):
    def __init__(self, n_features, n_classes, n_layers, n_units, p=0.5):
        super().__init__()
        self.p = float(p)
        if n_layers > 0:
            layers = [
                nn.Linear(n_features, n_units),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.p) if self.p > 0 else nn.Identity(),
            ]
            layers += [
                nn.Linear(n_units, n_units),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.p) if self.p > 0 else nn.Identity(),
            ] * (n_layers - 1)
            self.clf = nn.Sequential(
                *layers,
                nn.Linear(n_units, n_classes),
            )
        else:
            self.clf = nn.Linear(n_units, n_classes)

    def forward(self, x):
        return self.clf(x)


class simple_clf(nn.Module):
    def __init__(self, n_features, n_classes, p=0.5):
        super().__init__()
        self.p = p
        self.clf = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.clf(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryEntropyLoss_weight(nn.Module):
    def __init__(self, weight=None, size_average=True, is_weight=True):
        super(BinaryEntropyLoss_weight, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_weight = is_weight
        self.class_num = np.array(
            [
                [
                    12885,
                    1254,
                    3621,
                    1561 / 2,
                    1858,
                    2513 / 2,
                    1008 / 2,
                    2822,
                    53,
                    45,
                    28,
                    1093,
                    688,
                    537,
                    1066,
                    21,
                    530,
                    210,
                    902,
                    1482 / 2,
                    172,
                    3777 / 2,
                    802,
                    2965,
                    322,
                    8228 / 2,
                    328,
                    11,
                ]
            ]
        )
        self.class_num = np.power((1 - self.class_num / 30000), 2)

    def forward(self, input, target):
        self.weight = torch.cuda.FloatTensor(
            self.class_num.repeat(target.shape[0], axis=0)
        )
        loss = F.binary_cross_entropy(input, target, self.weight, self.size_average)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = (
            logit
            - logit * target
            + max_val
            + ((-max_val).exp() + (-logit - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


def balance_data(X_train, y_train):
    k = len(y_train)
    freq_per_class = y_train.mean(axis=0)
    balance_freqs_per_class = 1 / (freq_per_class)
    balance_freq_per_sample = y_train * balance_freqs_per_class
    balance_freq_per_sample = (
        balance_freq_per_sample.max(axis=1) / balance_freqs_per_class.max()
    )
    indices = choices(np.arange(k), weights=balance_freq_per_sample, k=k)
    return (X_train[indices], y_train[indices])


def get_scheduler(optim, total_steps, len_dl, args):
    lr = (
        float(args.lr) * (int(args.batch_size_per_gpu) * get_world_size()) / 256.0
    )  # linear scaling rule
    if args.schedule == "Constant":
        scheduler = ConstantLR(optim, float(lr), total_iters=0)
        wd_scheduler, lr_scheduler = None, None
    elif args.schedule == "OneCycle":
        scheduler = OneCycleLR(optim, float(lr), total_steps=int(total_steps))
        wd_scheduler, lr_scheduler = None, None
    elif args.schedule == "Cosine":
        scheduler = CosineAnnealingLR(optim, int(total_steps))
        wd_scheduler, lr_scheduler = None, None
    elif args.schedule == "Cyclic":
        scheduler = CyclicLR(
            optim,
            float(lr) / 25,
            float(lr),
            int(total_steps / 8),
            cycle_momentum=False,
        )
        wd_scheduler, lr_scheduler = None, None
    elif args.schedule == "DINO_cosine":
        lr_scheduler = cosine_scheduler(
            float(lr) / 256.0,  # linear scaling rule
            1e-6,
            args.epochs,
            len_dl,
            warmup_epochs=10,
        )
        wd_scheduler = cosine_scheduler(
            args.wd,
            args.wd_end,
            args.epochs,
            len_dl,
        )
        scheduler = None
    elif args.schedule == "Flat":
        scheduler = None
        wd_scheduler, lr_scheduler = None, None
    else:
        print(f"Scheduler {args.schedule} not implemented.")
        sys.exit()
    print(f"Using {args.schedule} learning rate schedule")

    return scheduler, lr_scheduler, wd_scheduler


def get_optimizer(parameters, args):
    lr = (
        float(args.lr) * (int(args.batch_size_per_gpu) * get_world_size()) / 256.0
    )  # linear scaling rule
    if args.optimizer == "AdamW":
        optim = torch.optim.AdamW(
            parameters,
            lr=float(lr),
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=float(args.wd),
            amsgrad=False,
        )
    elif args.optimizer == "SGD":
        optim = torch.optim.SGD(parameters, lr=float(lr), weight_decay=float(args.wd))

    else:
        print(f"Optimizer {args.optimizer} not implemented.")
        sys.exit()
    print(f"Using {args.optimizer} optimizer")
    return optim


def network_train(num_classes, X_train, y_train, X_test, y_test, config=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["classification"]["loss"] == "focal":
        criterion = FocalLoss()
        model = Multilabel_classifier(
            X_train.shape[1], y_train.shape[1], with_sigmoid=False
        ).to(device)
    else:
        criterion = nn.BCELoss()
        # criterion = torch.nn.BCEWithLogitsLoss()
        model = Multilabel_classifier(
            X_train.shape[1],
            y_train.shape[1],
            with_sigmoid=True,
            num_blocks=3,
            hidden_units=500,
        ).to(device)
    batch_size = config["classification"]["batch_size"]
    if config["classification"]["balance"]:
        # train_dataset = AutoBalancedPrecomputedFeatures(X_train.detach().cpu(), y_train.detach().cpu(), balance=True)
        # train_data_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     # sampler=torch.utils.data.RandomSampler(train_dataset, shuffle=True),
        #     batch_size=batch_size,
        #     num_workers=10,
        #     pin_memory=True,
        #     drop_last=True,
        # )
        X_train, y_train = balance_data(
            np.array(X_train.detach().cpu()), np.array(y_train.detach().cpu())
        )
        X_train = torch.Tensor(X_train).cuda()
        y_train = torch.Tensor(y_train).cuda()
    num_epochs = 100
    batches = [
        (X_train[i : i + batch_size, :], y_train[i : i + batch_size, :])
        for i in range(0, X_train.shape[0], batch_size)
    ]
    test_batches = [
        (X_test[i : i + batch_size, :], y_test[i : i + batch_size, :])
        for i in range(0, X_test.shape[0], batch_size)
    ]
    if batches[-1][0].shape[0] == 1:
        batches = batches[:-1]
    if test_batches[-1][0].shape[0] == 1:
        test_batches = test_batches[:-1]

    # total_steps = num_epochs * (len(train_dataset) / batch_size)
    total_steps = num_epochs * len(batches)
    # scheduler = OneCycleLR(optimizer, max_lr = 0.01, total_steps=total_steps)
    if config["classification"]["scheduler"] == "cycle":
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
        scheduler = CyclicLR(
            optimizer, 0.00001, 0.01, total_steps / 8, cycle_momentum=False
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5,
            amsgrad=False,
        )
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
    pbar = tqdm(range(num_epochs))
    # pbar = (range(num_epochs))
    running_loss = 0
    losses = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in pbar:
        temp_running_loss = 0
        model = model.train()
        for ind, (X_batch, y_batch) in enumerate(batches):
            # for ind, (X_batch, y_batch) in enumerate(train_data_loader):
            optimizer.zero_grad()
            # loss = criterion(model(X_batch), y_batch, reduction='mean')
            prediction = model(X_batch.to(device))
            loss = criterion(prediction, y_batch.to(device))
            loss.backward()
            temp_running_loss += loss.item()
            optimizer.step()
            scheduler.step()
        if (epoch % 10) == 0:
            model = model.eval()
            predictions = []
            targets = []
            for ind, (X_batch, y_batch) in enumerate(batches):
                prediction = model(X_batch.to(device))
                predictions.extend(prediction.detach().cpu().numpy())
                targets.extend(y_batch.detach().cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            train_loss = criterion(
                torch.Tensor(predictions), torch.Tensor(targets)
            ).item()
            train_score = f1_score(
                threshold_output(predictions).astype(int), targets, average="macro"
            )
            train_accuracies.append(train_score)
            train_losses.append(train_loss)
            predictions = []
            targets = []
            for ind, (X_batch, y_batch) in enumerate(test_batches):
                prediction = model(X_batch.to(device))
                predictions.extend(prediction.detach().cpu().numpy())
                targets.extend(y_batch.detach().cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            test_loss = criterion(
                torch.Tensor(predictions), torch.Tensor(targets)
            ).item()
            test_score = f1_score(
                threshold_output(predictions).astype(int), targets, average="macro"
            )
            test_accuracies.append(test_score)
            test_losses.append(test_loss)

        pbar.set_description(
            f"epoch: {epoch}, train loss {train_loss:.2f}, train score {train_score}, test loss {test_loss:.2f}, test score {test_score}"
        )

    np.save(
        f'{config["model"]["output_dir"]}/training_stats_network.npy',
        (train_losses, test_losses, train_accuracies, test_accuracies),
    )
    return model.eval()


def simpler_clf_train(num_classes, X_train, y_train, X_test, y_test, config=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["classification"]["loss"] == "focal":
        criterion = FocalLoss()
        model = simple_clf(X_train.shape[1], y_train.shape[1]).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss()
        model = simple_clf(X_train.shape[1], y_train.shape[1]).to(device)
    batch_size = config["classification"]["batch_size"]
    num_epochs = 100
    if config["classification"]["balance"]:
        X_train, y_train = balance_data(
            np.array(X_train.detach().cpu()), np.array(y_train.detach().cpu())
        )
        X_train = torch.Tensor(X_train).cuda()
        y_train = torch.Tensor(y_train).cuda()
    batches = [
        (X_train[i : i + batch_size, :], y_train[i : i + batch_size, :])
        for i in range(0, X_train.shape[0], batch_size)
    ]
    test_batches = [
        (X_test[i : i + batch_size, :], y_test[i : i + batch_size, :])
        for i in range(0, X_test.shape[0], batch_size)
    ]
    if batches[-1][0].shape[0] == 1:
        batches = batches[:-1]
    if test_batches[-1][0].shape[0] == 1:
        test_batches = test_batches[:-1]

    total_steps = num_epochs * len(batches)
    if config["classification"]["scheduler"] == "cycle":
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
        scheduler = CyclicLR(
            optimizer, 0.00001, 0.01, total_steps / 8, cycle_momentum=False
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5,
            amsgrad=False,
        )
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=total_steps)
    pbar = tqdm(range(num_epochs))
    # pbar = (range(num_epochs))
    running_loss = 0
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in pbar:
        model.train()
        temp_running_loss = 0
        for ind, (X_batch, y_batch) in enumerate(batches):
            optimizer.zero_grad()
            # loss = criterion(model(X_batch), y_batch, reduction='mean')
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            temp_running_loss += loss.item()
            optimizer.step()
            scheduler.step()
        if (epoch % 10) == 0:
            model = model.eval()
            predictions = []
            targets = []
            for ind, (X_batch, y_batch) in enumerate(batches):
                prediction = model(X_batch.to(device))
                predictions.extend(prediction.detach().cpu().numpy())
                targets.extend(y_batch.detach().cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            train_loss = criterion(
                torch.Tensor(predictions), torch.Tensor(targets)
            ).item()
            train_score = f1_score(
                threshold_output(predictions).astype(int), targets, average="macro"
            )
            train_accuracies.append(train_score)
            train_losses.append(train_loss)
            predictions = []
            targets = []
            for ind, (X_batch, y_batch) in enumerate(test_batches):
                prediction = model(X_batch.to(device))
                predictions.extend(prediction.detach().cpu().numpy())
                targets.extend(y_batch.detach().cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            test_loss = criterion(
                torch.Tensor(predictions), torch.Tensor(targets)
            ).item()
            test_score = f1_score(
                threshold_output(predictions).astype(int), targets, average="macro"
            )
            test_accuracies.append(test_score)
            test_losses.append(test_loss)

        pbar.set_description(
            f"epoch: {epoch}, train loss {train_loss:.2f}, train score {train_score}, test loss {test_loss:.2f}, test score {test_score}"
        )

    np.save(
        f'{config["model"]["output_dir"]}/training_stats_clf.npy',
        (train_losses, test_losses, train_accuracies, test_accuracies),
    )
    return model.eval()


def simpler_clf_predict(classifier, X_test):
    prediction = threshold_output(classifier(X_test)).int().cpu().detach().numpy()
    return prediction


def simpler_clf_save(classifier, config, task):
    torch.save(classifier.state_dict(), config["classification"][f"{task}_classifier"])


def simpler_clf_load(config, task):
    return torch.load(config["classification"][f"{task}_classifier"])


def network_predict(classifier, X_test):
    prediction = threshold_output(classifier(X_test)).int().cpu().detach().numpy()
    return prediction


def network_save(classifier, config, task):
    torch.save(classifier.state_dict(), config["classification"][f"{task}_classifier"])


def network_load(config, task):
    return torch.load(config["classification"][f"{task}_classifier"])


def MLLR_train(num_classes, X_train, y_train, config=None):
    classifier = MLLR(max_iter=1000, num_classes=num_classes)
    classifier.fit(X_train, y_train)
    return classifier


def MLLR_predict(classifier, X_test):
    prediction = classifier.predict(X_test)
    return prediction


def MLLR_save(classifier, config, task):
    torch.save(classifier, config["classification"][f"{task}_classifier"])


def MLLR_load(config, task):
    return torch.load(config["classification"][f"{task}_classifier"])


def profile_features(X, y, inds_per_ID, IDs):
    new_X = []
    new_y = []
    for ID in IDs:
        new_X.append(X[inds_per_ID[ID]].mean(axis=0))
        new_y.append(y[inds_per_ID[ID]][0])
    return torch.Tensor(new_X), torch.Tensor(new_y)


def get_dataset(config, dataset):
    features_list = config["classification"][dataset]
    # all_features, cell_types, protein_locations, IDs = torch.load(features_list[0])
    all_features, protein_locations, cell_types, IDs = torch.load(features_list[0])
    if len(features_list) > 1:
        for features_path in features_list[1:]:
            (
                all_features_other,
                protein_locations_other,
                cell_types_other,
                IDs_other,
            ) = torch.load(features_path)
            all_features = torch.cat((all_features, all_features_other), axis=1)
            protein_locations.extend(protein_locations_other)
            cell_types.extend(cell_types_other)
            IDs.extend(IDs_other)
    return all_features, protein_locations, cell_types, IDs


class FocalBCELoss(torch.nn.Module):
    y_int = True  # y interpolation

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inp, targ):
        "Applies focal loss based on https://arxiv.org/pdf/1708.02002.pdf"
        ce_loss = F.binary_cross_entropy_with_logits(
            inp, targ, weight=self.weight, reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def write_to_tensorboard(writer, description_dict, t="epoch"):
    for k, v in description_dict.items():
        if k == "epoch":
            continue
        if k == "step":
            continue
        writer.add_scalar(k, v, description_dict[t])


def get_classifier(args, embed_dim):
    args.num_classes = int(args.num_classes)
    if args.classifier_type == "simple_clf":
        classifier = simple_clf(embed_dim, args.num_classes, p=float(args.dropout))
    elif args.classifier_type == "expanding_clf":
        classifier = expanding_clf(embed_dim, args.num_classes)
    elif args.classifier_type == "residual_clf":
        classifier = residual_clf(
            embed_dim,
            args.num_classes,
            n_layers=int(args.n_layers),
            norm_layer=args.norm_layer if "norm_layer" in args else None,
        )
    elif args.classifier_type == "residual_add_clf":
        classifier = residual_add_clf(
            embed_dim,
            args.num_classes,
            n_layers=int(args.n_layers),
            n_units=int(args.n_units),
            norm_layer=args.norm_layer if "norm_layer" in args else None,
            skip=args.skip,
        )
    elif args.classifier_type == "prototyping_clf":
        classifier = prototyping_clf(
            n_features=embed_dim,
            n_classes=int(args.num_classes),
            p=float(args.dropout),
            n_units=int(args.n_units),
            n_layers=int(args.n_layers),
        )
    elif args.classifier_type == "network":
        classifier = Multilabel_classifier(
            embed_dim,
            args.num_classes,
            with_sigmoid=False,
            num_blocks=1,
            hidden_units=256,
        )
    return classifier


def scale(features, features_mean=None, features_std=None):
    if features_mean is None:
        features_mean = features.mean(axis=0)
    if features_std is None:
        features_std = features.std(axis=0)
    transformed_features = (features - features_mean) / (features_std + 0.00001)
    return transformed_features, features_mean, features_std


def get_embeddings(features_to_fit, features_to_transform):
    scaled_features_to_fit, features_mean, features_std = scale(features_to_fit)
    scaled_features_to_transform, features_mean, features_std = scale(
        features_to_transform, features_mean, features_std
    )
    umap_unique = cuml.UMAP(
        init="spectral",
        metric="euclidean",
        min_dist=0.1,
        n_neighbors=15,
        n_components=2,
        spread=4,
        output_type="numpy",
        verbose=True,
        n_epochs=2600,
        transform_queue_size=20,
        random_state=42,
    )
    umap_unique = umap_unique.fit(scaled_features_to_fit.numpy())
    transformed_features = umap_unique.transform(scaled_features_to_transform.numpy())
    return (
        umap_unique,
        features_mean,
        features_std,
        transformed_features,
        scaled_features_to_transform,
    )


def plot_UMAP(df, labels, embedding, title):
    mat, labels = get_col_matrix(df, labels)
    plt.figure()
    plt.plot(np.nan, np.nan)
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.axis(False)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=0.1, label="All data", color="grey")
    for i in range(mat.shape[1]):
        indices = np.where((mat[:, i] == 1) & (mat[:, :].sum(axis=1) == 1))[0]
        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            s=0.1,
            label=labels[i],
            color=cmap(i / mat.shape[1]),
        )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(title, fontsize=15)
    lgnd = plt.legend(bbox_to_anchor=(1, 1), frameon=False)
    for h in lgnd.legendHandles:
        h._sizes = [30]


def get_col_matrix(df, labels):
    if len(labels) == 1:
        values = df[labels[0]]
        unique_values = sorted(np.unique(values))
        mat = np.zeros((len(df), len(unique_values)))
        for ind, value in enumerate(unique_values):
            mat[np.where(values == value)[0], ind] = 1
        columns = unique_values
    else:
        mat = df[sorted(labels)].values.astype(int)
        columns = sorted(labels)
    return mat, columns


def get_averaged_features(df, features, labels, sort=True):
    mat, columns = get_col_matrix(df, labels)
    averaged_features = []
    for key in range(len(columns)):
        indices = np.where(mat[:, key])
        averaged_features.append(features[indices].mean(axis=0))
    averaged_features = torch.stack(averaged_features)
    return averaged_features, columns


def get_heirarchical_clustering(
    df, features, labels, zero_diagonal=True, metric="similarity"
):
    averaged_features, columns = get_averaged_features(df, features, labels)
    if metric == "similarity":
        distance_matrix = cosine_similarity(averaged_features, averaged_features)
    else:
        distance_matrix = cosine_distances(averaged_features, averaged_features)

    Z = linkage(distance_matrix)
    plt.figure(figsize=(20, 5))
    dn = dendrogram(Z, labels=columns)
    plt.xticks(rotation=45)
    distance_matrix = distance_matrix[dn["leaves"], :][:, dn["leaves"]]
    columns = dn["ivl"]
    plt.figure(figsize=(10, 10))
    if zero_diagonal:
        for i in range(len(distance_matrix)):
            distance_matrix[i, i] = 0
    plt.imshow(distance_matrix)
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    return distance_matrix, columns, Z


def get_heterogeneousity_per_whole_image(
    df, features, indices, metric="distance", verbose=False
):
    cells_per_ID = df.iloc[indices].groupby("ID").groups
    mean_distances_per_ID = []
    for ID in tqdm(sorted(cells_per_ID.keys()), disable=verbose == False):
        if metric == "distance":
            distance_matrix = cosine_distances(
                features[cells_per_ID[ID]], features[cells_per_ID[ID]]
            )
            distances = np.triu(distance_matrix)[
                np.triu_indices(distance_matrix.shape[0], k=1)
            ]
            if len(distances) == 0:
                continue
            mean_distances_per_ID.append(distances.mean())
        elif metric == "std":
            mean_distances_per_ID.append(
                features[cells_per_ID[ID]].std(axis=0).mean().item()
            )
    return mean_distances_per_ID, sorted(cells_per_ID.keys())


def get_heterogeneousity_per_gene(
    merged_df, features, indices, metric="distance", verbose=False
):
    cells_per_gene = merged_df.iloc[indices].groupby(["Gene"]).groups
    keys = sorted(cells_per_gene.keys())
    mean_distances_per_ID = []
    for gene in tqdm(keys):
        if metric == "distance":
            distance_matrix = cosine_distances(
                features[cells_per_gene[gene]], features[cells_per_gene[gene]]
            )
            distances = np.triu(distance_matrix)[
                np.triu_indices(distance_matrix.shape[0], k=1)
            ]
            if len(distances) == 0:
                continue
            mean_distances_per_ID.append(distances.mean())
        elif metric == "std":
            mean_distances_per_ID.append(
                features[cells_per_gene[gene]].std(axis=0).mean().item()
            )
    return mean_distances_per_ID, keys


def get_heterogeneity_df(config_file):
    config = yaml.safe_load(open(config_file, "r"))
    df = pd.read_csv(config["embedding"]["df_path"])
    df_master = pd.read_csv("/scr/mdoron/Dino4Cells/Master_scKaggle.csv")
    df = pd.merge(df, df_master[["ID", "gene"]], on="ID")
    df["Gene"] = df["gene"]
    proto_heterogeneity_df = pd.read_csv(
        "/scr/mdoron/Dino4Cells/data/gene_heterogeneity.tsv", delimiter="\t"
    )
    proto_heterogeneity_df["v_spatial"] = proto_heterogeneity_df[
        "Single-cell variation spatial"
    ]
    proto_heterogeneity_df["v_intensity"] = proto_heterogeneity_df[
        "Single-cell variation intensity"
    ]
    proto_heterogeneity_df["HPA_variable"] = np.nan
    proto_heterogeneity_df.loc[
        (pd.isnull(proto_heterogeneity_df["v_intensity"]) == False), "v_intensity"
    ] = True
    proto_heterogeneity_df.loc[
        (pd.isnull(proto_heterogeneity_df["v_intensity"])), "v_intensity"
    ] = False
    proto_heterogeneity_df.loc[
        (pd.isnull(proto_heterogeneity_df["v_spatial"]) == False), "v_spatial"
    ] = True
    proto_heterogeneity_df.loc[
        (pd.isnull(proto_heterogeneity_df["v_spatial"])), "v_spatial"
    ] = False
    proto_heterogeneity_df.loc[
        (proto_heterogeneity_df["v_spatial"]) | (proto_heterogeneity_df["v_intensity"]),
        "HPA_variable",
    ] = True
    heterogeneity_df = pd.merge(proto_heterogeneity_df, df, on="Gene")[
        ["Gene", "ID", "v_spatial", "v_intensity", "HPA_variable", "cell_type"]
    ]
    return heterogeneity_df


def find_gene_enrichment(
    config_file,
    heterogenous_type="HPA_variable",
    granularity="gene",
    condition=None,
    use_embedding=False,
):
    config = yaml.safe_load(open(config_file, "r"))
    df = pd.read_csv(config["embedding"]["df_path"])
    heterogeneity_df = get_heterogeneity_df(config_file)
    merged_df = pd.merge(heterogeneity_df, df, on="ID", how="right").drop_duplicates()
    merged_df.loc[pd.isnull(merged_df["Gene"]), "Gene"] = ""
    merged_df = merged_df.reset_index()
    merged_df["original_index"] = merged_df.index.values
    merged_df["cell_type"] = merged_df.cell_type_x
    original_indices = merged_df.original_index.values
    gene_to_ind = merged_df[["Gene", "original_index"]].groupby("Gene").groups
    gene_to_ind = {k: original_indices[gene_to_ind[k]] for k in gene_to_ind.keys()}
    merged_df[pd.isnull(merged_df.Gene)].Gene = ""
    features, protein_localizations, cell_lines, IDs = torch.load(
        config["embedding"]["output_path"]
    )

    if condition == "U-2 OS":
        features = features[np.where(merged_df.cell_type == "U-2 OS")[0]]
        merged_df = merged_df[merged_df.cell_type == "U-2 OS"].reset_index()
    else:
        merged_df = merged_df.reset_index()

    if use_embedding:
        (
            umap_reducer,
            features_mean,
            features_std,
            embedding,
            scaled_features,
        ) = get_embeddings(torch.Tensor(features), torch.Tensor(features))
        scaled_features = embedding
    else:
        scaled_features, features_mean, features_std = scale(features)
        scaled_features = scaled_features.numpy()
    indices = merged_df.index.values
    merged_df["original_index"] = merged_df.index.values

    if granularity == "gene":
        mean_stds, sorted_genes = get_heterogeneousity_per_gene(
            merged_df, scaled_features, indices, metric="std", verbose=False
        )
    elif granularity == "image":
        mean_stds, sorted_images = get_heterogeneousity_per_whole_image(
            merged_df, scaled_features, indices, metric="std", verbose=False
        )

    metric = mean_stds
    sorted_distances = np.array(metric)[np.argsort(metric)[::-1]]
    if granularity == "gene":
        genes_sorted = np.array(sorted_genes)[np.argsort(metric)[::-1]]
        enrichment_raw = (
            pd.DataFrame(genes_sorted)[0]
            .isin(merged_df[merged_df[heterogenous_type] == True].Gene.unique())
            .values
        )
        enrichment = []
        for ind in tqdm(range(1, len(genes_sorted), 1)):
            enrichment.append(enrichment_raw[:ind].mean())
        images_sorted = None
        sorted_distances_per_image = None
        sorted_distances_per_gene = sorted_distances
    elif granularity == "image":
        images_sorted = np.array(sorted_images)[np.argsort(metric)[::-1]]
        merged_df["image_rank"] = merged_df.ID.map(
            dict(zip(images_sorted, range(len(images_sorted))))
        )
        merged_df["image_score"] = merged_df.ID.map(
            dict(zip(images_sorted, sorted_distances))
        )
        genes_and_image_ranks = merged_df.groupby("Gene")["image_rank"].min()
        genes_and_image_scores = merged_df.groupby("Gene")["image_score"].max()
        genes_ranked_by_images_scores = genes_and_image_scores.values
        genes_ranked_by_images_rank = genes_and_image_ranks.values
        genes_ranked_by_images_names = genes_and_image_ranks.index.values
        genes_sorted = genes_ranked_by_images_names[
            np.argsort(genes_ranked_by_images_rank)
        ]

        enrichment_raw = (
            pd.DataFrame(genes_sorted)[0]
            .isin(merged_df[merged_df[heterogenous_type] == True].Gene.unique())
            .values
        )
        enrichment = []
        for ind in tqdm(range(1, len(genes_sorted), 1)):
            enrichment.append(enrichment_raw[:ind].mean())
        sorted_distances_per_image = sorted_distances
        sorted_distances_per_gene = np.array(genes_ranked_by_images_scores)[
            np.argsort(genes_ranked_by_images_scores)[::-1]
        ]

    return (
        enrichment,
        merged_df,
        genes_sorted,
        images_sorted,
        sorted_distances_per_gene,
        sorted_distances_per_image,
    )


def explore_single_gene(
    features,
    protein_localizations,
    merged_df,
    embedding,
    genes_sorted,
    gene_rank,
    number_of_clusters=2,
    color_by="whole_image",
    cell_preds=None,
):
    gene2uniprot = pd.read_csv("XML_df/HPA_img_df_new.csv")[
        ["gene", "uniport_id"]
    ].drop_duplicates()
    uniprot2name = pd.read_csv("uniport_interactions.tsv", delimiter="\t")[
        ["Entry", "Gene Names"]
    ].drop_duplicates()
    gene2name = pd.merge(
        uniprot2name, gene2uniprot, right_on="uniport_id", left_on="Entry"
    )[["gene", "Gene Names"]]
    t = number_of_clusters
    rank = gene_rank
    merged_df["rank"] = merged_df.Gene.map(
        dict(zip(genes_sorted, range(len(genes_sorted))))
    )
    gene = genes_sorted[rank]
    gane_name = gene2name[gene2name.gene == gene]["Gene Names"].iloc[0].split(" ")[0]
    gene_indices = np.where(merged_df.Gene == gene)[0]
    gene_df = merged_df.iloc[gene_indices].reset_index()
    targets = torch.Tensor(protein_localizations)
    if color_by == "whole_image":
        gene_df["well"] = gene_df.ID
        color_by_name = "Image ID"
    elif color_by == "plate":
        gene_df["well"] = gene_df.ID.apply(lambda x: x.split("_")[0])
        color_by_name = "Plate"
    elif color_by == "well":
        gene_df["well"] = gene_df.ID.apply(lambda x: x.split("_")[1])
        color_by_name = "Well"
    dists = pdist(features[gene_indices], metric="euclidean")
    Z = linkage(dists, "ward")
    clusters = fcluster(Z, criterion="maxclust", t=t)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor="white")
    plt.scatter(np.NaN, np.NaN)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor="white")
    plt.title(
        f'UMAP embedding of all single cells stained for gene {gane_name}\nrank: {rank + 1}, ground truth: {"heterogeneous" if gene_df.HPA_variable.iloc[0] == True else "unlabeled"}'
    )

    channels = [0]
    markers = ["*", "^", "o", "|", "v", "s"]
    wells = np.unique(gene_df.well)
    colors = {wells[i]: "rgbkymrgbkymrgbkym"[i] for i in range(len(wells))}
    markers = {i: markers[i - 1] for i in np.unique(clusters)}

    for label in np.unique(clusters):
        indices = np.where(clusters == label)[0]
        gene_df.loc[indices, "cluster"] = label

    inds = []
    embedding, _ = get_embeddings(features[gene_indices])
    for cluster in range(1, max(clusters) + 1):
        indices = np.where(clusters == cluster)[0]
        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            #                     marker=markers[cluster],
            c=[colors[i] for i in gene_df.iloc[indices].well],
            label=f"Cluster {cluster}",
        )

    ax2 = ax.twinx()
    for w in wells:
        ax2.scatter(np.NaN, np.NaN, color=colors[w], label=f"Image {w}", marker="o")
    ax2.get_yaxis().set_visible(False)

    #     lgnd = ax.legend(frameon=False, scatterpoints=1, fontsize=10, loc='upper left', title='Raw feature clustering')
    #     for h in lgnd.legendHandles: h._original_facecolor = 'black'
    ax2.legend(frameon=False, loc="lower right", title="Image IDs")

    ax.axis("off")
    ax2.axis("off")
    plt.tight_layout()

    row_colors1 = gene_df["well"].map(colors)
    row_colors1.name = color_by_name

    b = ((targets[gene_indices]).numpy()).astype(int)
    c2 = [str(list(i)) for i in b]
    lut1 = dict(
        zip(
            np.unique(c2),
            [cmap(i / len(np.unique(c2))) for i in range(len(np.unique(c2)))],
        )
    )
    series = pd.DataFrame(c2)[0]
    series.name = "Labels"
    row_colors3 = series.map(lut1)
    row_colors = pd.concat([row_colors1, row_colors3], axis=1)

    if cell_preds is not None:
        b = ((cell_preds[gene_indices])).astype(int)
        c1 = [str(list(i)) for i in b]
        series = pd.DataFrame(c1)[0]
        series.name = "Prediction"
        lut1 = dict(
            zip(
                np.unique(c1),
                [cmap(i / len(np.unique(c1))) for i in range(len(np.unique(c1)))],
            )
        )
        row_colors2 = series.map(lut1)
        row_colors = pd.concat([row_colors1, row_colors2, row_colors3], axis=1)

    c = clustermap(
        pd.DataFrame(squareform(dists)),
        figsize=(7.5, 7.5),
        cbar_pos=None,
        xticklabels=[],
        yticklabels=[],
        cmap="Blues",
        dendrogram_ratio=[0, 0],
        col_cluster=True,
        row_cluster=True,
        row_colors=row_colors,
        cbar_kws={"size": 20},
        facecolor="white",
    )
    # c.ax_row_colors.set_xlabel('Well', fontsize=18, rotation=90)
    num_to_protein = {
        protein_to_num_single_cells[k]: k for k in protein_to_num_single_cells.keys()
    }
    for cluster in np.unique(clusters):
        plt.figure(figsize=(5, 5), facecolor="white")
        plt.imshow(
            io.imread(gene_df[(gene_df.cluster == cluster)].file.iloc[0])[
                :, :, channels
            ],
            cmap="Greys_r",
            vmin=0,
            vmax=255,
        )
        plt.grid(None)
        plt.axis("off")
        plt.title(
            f'cluster {cluster}, {", ".join([num_to_protein[i] for i in np.where(targets[gene_indices][np.where(gene_df.cluster == cluster)[0][0]])[0]])}'
        )

    for ID in gene_df.ID.unique():
        gene_type = (
            "heterogeneous"
            if merged_df[merged_df.ID == ID].HPA_variable.iloc[0] == True
            else "unlabeled"
        )
        img = io.imread(f"/scr/mdoron/Dino4Cells/data/whole_images/{ID}.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img[:, :, [0, 2, 3]])
        axes[0].axis(False)
        axes[1].imshow(img[:, :, [1]], cmap="Greys_r")
        axes[1].axis(False)
        axes[0].set_title("Non protein channels")
        axes[1].set_title("Protein channel")
        plt.tight_layout()


def get_gene_heterogeneity_enrichement(
    config_files, heterogenous_type="HPA_variable", condition=None, use_embedding=False
):
    enrichments_conditioned_on_gene = []
    enrichments_conditioned_on_image = []
    genes_sorted_conditioned_on_image = []
    genes_sorted_conditioned_on_gene = []
    merged_dfs = []
    sorted_metrics_conditioned_on_image = []
    sorted_metrics_conditioned_on_gene = []
    images_sorted_conditioned_on_image = []
    average_precisions_condition_on_gene = []
    average_precisions_condition_on_image = []
    for config_file in config_files:
        (
            enrichment,
            merged_df,
            genes_sorted,
            _,
            sorted_metric_conditioned_on_gene,
            _,
        ) = find_gene_enrichment(
            config_file,
            heterogenous_type=heterogenous_type,
            condition=condition,
            use_embedding=use_embedding,
            granularity="gene",
        )
        sorted_metrics_conditioned_on_gene.append(sorted_metric_conditioned_on_gene)
        genes_sorted_conditioned_on_gene.append(genes_sorted)
        enrichments_conditioned_on_gene.append(enrichment)
        gene_ground_truth = (
            pd.merge(
                pd.DataFrame(genes_sorted, columns=["Gene"]),
                merged_df[["Gene", "HPA_variable"]].drop_duplicates(),
                on="Gene",
            )["HPA_variable"]
            .apply(lambda x: pd.isnull(x) == False)
            .astype(int)
            .values
        )
        gene_predictions = np.where(
            np.isnan(sorted_metric_conditioned_on_gene),
            0,
            sorted_metric_conditioned_on_gene,
        )
        average_precisions_condition_on_gene.append(
            average_precision_score(gene_ground_truth, gene_predictions)
        )

        (
            enrichment,
            merged_df,
            genes_sorted,
            images_sorted,
            sorted_metric_conditioned_on_gene,
            sorted_metric_conditioned_on_image,
        ) = find_gene_enrichment(
            config_file,
            heterogenous_type=heterogenous_type,
            condition=condition,
            use_embedding=use_embedding,
            granularity="image",
        )
        sorted_metrics_conditioned_on_image.append(sorted_metric_conditioned_on_gene)
        genes_sorted_conditioned_on_image.append(genes_sorted)
        enrichments_conditioned_on_image.append(enrichment)
        gene_ground_truth = (
            pd.merge(
                pd.DataFrame(genes_sorted, columns=["Gene"]),
                merged_df[["Gene", "HPA_variable"]].drop_duplicates(),
                on="Gene",
            )["HPA_variable"]
            .apply(lambda x: pd.isnull(x) == False)
            .astype(int)
            .values
        )
        gene_predictions = np.where(
            np.isnan(sorted_metric_conditioned_on_gene),
            0,
            sorted_metric_conditioned_on_gene,
        )
        average_precisions_condition_on_image.append(
            average_precision_score(gene_ground_truth, gene_predictions)
        )

        merged_dfs.append(merged_df)
        images_sorted_conditioned_on_image.append(images_sorted)
    return (
        enrichments_conditioned_on_gene,
        enrichments_conditioned_on_image,
        merged_dfs,
        genes_sorted_conditioned_on_gene,
        genes_sorted_conditioned_on_image,
        sorted_metrics_conditioned_on_gene,
        sorted_metrics_conditioned_on_image,
        images_sorted_conditioned_on_image,
        average_precisions_condition_on_gene,
        average_precisions_condition_on_image,
    )


def plot_gene_heterogeneity_enrichement(
    labels, enrichments, enrichments_image, heterogenous_type
):
    if len(enrichments) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="white")
        for ind, label in enumerate(labels):
            #             plt.plot(enrichments[ind], label=label)
            plt.plot(enrichments[ind][:1000], label=label)
        plt.legend(frameon=False)
        ax.set_xlabel("Number of genes, sorted by feature variance")
        ax.set_ylabel("Percentage of ground truth heterogeneous genes")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xticks(range(0, 1001, 100))
        print("")
        plt.title(f"Heterogeneous gene ranking ({heterogenous_type})")
    if len(enrichments_image) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="white")
        for ind, label in enumerate(labels):
            #             plt.plot(enrichments_image[ind], label=label)
            plt.plot(enrichments_image[ind][:1000], label=label)
        plt.legend(frameon=False)
        ax.set_xlabel("Number of genes, sorted by image heterogeneity")
        ax.set_ylabel("Percentage of ground truth heterogeneous genes")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xticks(range(0, 1001, 100))
        print("")
        plt.title(f"Heterogeneous gene ranking ({heterogenous_type})")


def generate_all_figures(config_file, labels=protein_to_num_full, feature_type="full"):
    config_file = config_file
    config = yaml.safe_load(open(config_file, "r"))
    df = pd.read_csv(config["embedding"]["df_path"])
    if feature_type == "full":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
    elif feature_type == "average":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
        mean_df = pd.DataFrame(
            pd.concat(
                (
                    pd.DataFrame(
                        features, columns=[f"f_{i}" for i in range(len(features[0]))]
                    ),
                    pd.DataFrame(IDs, columns=["ID"]),
                ),
                axis=1,
            )
        ).groupby("ID")
        mean_features = mean_df.mean()[
            [f"f_{i}" for i in range(len(features[0]))]
        ].values
        features = torch.Tensor(mean_features)
        df = df.groupby("ID").first().reset_index()
    embedding, scaled_features = get_embeddings(features)
    heterogeneity_df = get_heterogeneity_df(config_file)
    heterogeneity_key = "v_spatial"
    heterogeneous_indices = np.where(
        df.ID.isin(heterogeneity_df[heterogeneity_df[heterogeneity_key]].ID)
    )[0]
    homogeneous_indices = np.where(
        df.ID.isin(heterogeneity_df[heterogeneity_df[heterogeneity_key]].ID) == False
    )[0]
    mean_distances_hetero = get_heterogeneousity_per_whole_image(
        df, scaled_features, heterogeneous_indices, metric="distance", verbose=False
    )
    mean_distances_homo = get_heterogeneousity_per_whole_image(
        df, scaled_features, homogeneous_indices, metric="distance", verbose=False
    )
    mean_stds_hetero = get_heterogeneousity_per_whole_image(
        df, scaled_features, heterogeneous_indices, metric="std", verbose=False
    )
    mean_stds_homo = get_heterogeneousity_per_whole_image(
        df, scaled_features, homogeneous_indices, metric="std", verbose=False
    )
    distance_matrix, columns, Z = get_heirarchical_clustering(
        df, features, ["cell_type"]
    )
    distance_matrix, columns, Z = get_heirarchical_clustering(df, features, labels)
    plot_UMAP(df, ["cell_type"], embedding, "Cell line")
    plot_UMAP(df, labels, embedding, "Protein localization")
    (
        enrichments_conditioned_on_gene,
        enrichments_conditioned_on_image,
        merged_dfs,
        genes_sorted_conditioned_on_gene,
        genes_sorted_conditioned_on_image,
        sorted_metrics_conditioned_on_gene,
        sorted_metrics_conditioned_on_image,
        images_sorted_conditioned_on_image,
        average_precisions_condition_on_gene,
        average_precisions_condition_on_image,
    ) = get_gene_heterogeneity_enrichement(heterogenous_type="HPA_variable")
    plot_gene_heterogeneity_enrichement(
        enrichments_conditioned_on_gene,
        enrichments_conditioned_on_image,
        heterogenous_type="HPA_variable",
    )


def create_protein_hierarchy(
    config_file, label_type, label, feature_type="full", apply_pca=True
):
    if label_type == "single_cells":
        labels = protein_to_num_single_cells
        low_level_labels = label_dict.hierarchical_organization_single_cell_low_level
        high_level_labels = label_dict.hierarchical_organization_single_cell_high_level
    elif label_type == "whole_image":
        labels = protein_to_num_full
        low_level_labels = label_dict.hierarchical_organization_whole_image_low_level
        high_level_labels = label_dict.hierarchical_organization_whole_image_high_level

    config = yaml.safe_load(open(config_file, "r"))
    df = pd.read_csv(config["embedding"]["df_path"])
    if feature_type == "full":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
    elif feature_type == "average":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
        mean_df = pd.DataFrame(
            pd.concat(
                (
                    pd.DataFrame(
                        features, columns=[f"f_{i}" for i in range(len(features[0]))]
                    ),
                    pd.DataFrame(IDs, columns=["ID"]),
                ),
                axis=1,
            )
        ).groupby("ID")
        mean_features = mean_df.mean()[
            [f"f_{i}" for i in range(len(features[0]))]
        ].values
        features = torch.Tensor(mean_features)
        df = df.groupby("ID").first().reset_index()
    elif feature_type == "max":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
        max_df = pd.DataFrame(
            pd.concat(
                (
                    pd.DataFrame(
                        features, columns=[f"f_{i}" for i in range(len(features[0]))]
                    ),
                    pd.DataFrame(IDs, columns=["ID"]),
                ),
                axis=1,
            )
        ).groupby("ID")
        max_features = max_df.max()[[f"f_{i}" for i in range(len(features[0]))]].values
        features = torch.Tensor(max_features)
        df = df.groupby("ID").first().reset_index()
    else:
        print(f"{feature_type} is not a known feature type, try again!")
        return

    scaled_features, features_mean, features_std = scale(features)
    protein_averaged_features, protein_columns = get_averaged_features(
        df, torch.Tensor(scaled_features), labels
    )
    protein_distance_matrix = 1 - squareform(
        pdist(protein_averaged_features, metric="cosine")
    )

    scaled_averaged_features, features_mean, features_std = scale(
        protein_averaged_features
    )
    if apply_pca:
        pca = decomposition.PCA(n_components=10)
        protein_averaged_features = pca.fit_transform(protein_averaged_features)
    protein_distance_matrix = 1 - squareform(
        pdist(protein_averaged_features, metric="cosine")
    )

    ground_truth_distance_matrix = np.zeros(protein_distance_matrix.shape)
    for g in high_level_labels:
        inner_indices = []
        for l in g:
            index = np.where(np.array(protein_columns) == l)[0]
            if len(index) > 0:
                inner_indices.append(index[0])
        for i in inner_indices:
            for j in inner_indices:
                ground_truth_distance_matrix[i][j] = 0.5
    for g in low_level_labels:
        inner_indices = []
        for l in g:
            index = np.where(np.array(protein_columns) == l)[0]
            if len(index) > 0:
                inner_indices.append(index[0])
        for i in inner_indices:
            for j in inner_indices:
                ground_truth_distance_matrix[i][j] = 1
    ground_truth_Z = linkage(ground_truth_distance_matrix)

    Z = linkage(protein_distance_matrix)
    dflt_col = "#808080"
    D_leaf_colors = {}

    colors = [cmap(i) for i in np.linspace(0, 1, len(low_level_labels))]
    for ind, g in enumerate(low_level_labels):
        for l in g:
            if l in protein_columns:
                D_leaf_colors[l] = colors[ind][:3]

    link_cols = {}
    for i, _ in enumerate(Z[:, :2].astype(int)):
        link_cols[i + 1 + len(Z)] = dflt_col

    # Dendrogram
    fig, ax = plt.subplots(figsize=(15, 2.5))
    D = dendrogram(
        Z=Z,
        labels=sorted(protein_columns),
        color_threshold=None,
        leaf_font_size=12,
        leaf_rotation=90,
        link_color_func=lambda x: link_cols[x],
    )
    plt.yticks([])
    for ind, l in enumerate(D["ivl"]):
        ax.get_xticklabels()[ind].set_color(D_leaf_colors[l])

    plt.figure()
    dn = dendrogram(ground_truth_Z, labels=protein_columns)
    ground_truth_distance_matrix = ground_truth_distance_matrix[dn["leaves"], :][
        :, dn["leaves"]
    ]
    protein_distance_matrix = protein_distance_matrix[dn["leaves"], :][:, dn["leaves"]]
    ground_truth_Z = linkage(ground_truth_distance_matrix)
    high_level_clusters = fcluster(ground_truth_Z, criterion="distance", t=2)
    low_level_clusters = fcluster(ground_truth_Z, criterion="distance", t=0.75)
    high_level_groups = pd.DataFrame(high_level_clusters).groupby(0).groups
    low_level_groups = pd.DataFrame(low_level_clusters).groupby(0).groups
    high_level_groups = {
        k: [high_level_groups[k].min(), high_level_groups[k].max()]
        for k in high_level_groups.keys()
    }
    low_level_groups = {
        k: [low_level_groups[k].min(), low_level_groups[k].max()]
        for k in low_level_groups.keys()
    }

    columns = dn["ivl"]
    plt.figure(figsize=(10, 10))
    plt.imshow(ground_truth_distance_matrix, cmap="Blues")
    plt.title(f"{label} - Protein ground truth matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print("")

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(protein_distance_matrix, cmap="Blues")
    plt.title(f"{label} - DINO similarity matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    for r in high_level_groups.keys():
        min_index, max_index = high_level_groups[r]
        rect = patches.Rectangle(
            (min_index - 0.5, min_index - 0.5),
            max_index - min_index + 1,
            max_index - min_index + 1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    for r in low_level_groups.keys():
        min_index, max_index = low_level_groups[r]
        rect = patches.Rectangle(
            (min_index - 0.5, min_index - 0.5),
            max_index - min_index + 1,
            max_index - min_index + 1,
            linewidth=2,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect)
    print("")

    for i in range(len(protein_distance_matrix)):
        protein_distance_matrix[i, i] = 0
        ground_truth_distance_matrix[i, i] = 0

    mantel_test = mantel.test(
        squareform(protein_distance_matrix), squareform(ground_truth_distance_matrix)
    )
    print(f"{label}, protein hierarchy mantel test: {mantel_test}")
    protein_hierarchy_mantel_test = mantel_test[0]
    return protein_hierarchy_mantel_test


def create_cell_comparison(config_file, label, apply_pca=True, feature_type="full"):
    config = yaml.safe_load(open(config_file, "r"))
    df = pd.read_csv(config["embedding"]["df_path"])
    if feature_type == "full":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
    elif feature_type == "average":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
        mean_df = pd.DataFrame(
            pd.concat(
                (
                    pd.DataFrame(
                        features, columns=[f"f_{i}" for i in range(len(features[0]))]
                    ),
                    pd.DataFrame(IDs, columns=["ID"]),
                ),
                axis=1,
            )
        ).groupby("ID")
        mean_features = mean_df.mean()[
            [f"f_{i}" for i in range(len(features[0]))]
        ].values
        features = torch.Tensor(mean_features)
        df = df.groupby("ID").first().reset_index()
    elif feature_type == "max":
        features, protein_localizations, cell_lines, IDs = torch.load(
            config["embedding"]["output_path"]
        )
        max_df = pd.DataFrame(
            pd.concat(
                (
                    pd.DataFrame(
                        features, columns=[f"f_{i}" for i in range(len(features[0]))]
                    ),
                    pd.DataFrame(IDs, columns=["ID"]),
                ),
                axis=1,
            )
        ).groupby("ID")
        max_features = max_df.max()[[f"f_{i}" for i in range(len(features[0]))]].values
        features = torch.Tensor(max_features)
        df = df.groupby("ID").first().reset_index()

    averaged_features, cell_columns = get_averaged_features(
        df, features, ["cell_type"], sort=True
    )
    path_to_rna = "/scr/mdoron/Dino4Cells/rna_cellline.tsv"
    rna = pd.read_csv(path_to_rna, delimiter="\t")
    tpms = []
    for cell_line in cell_columns:
        df = rna[rna["Cell line"] == cell_line]
        tpms.append(df.TPM.to_numpy())
    tpms = np.array(tpms)
    scaled_tpms, features_mean, features_std = scale(tpms)
    scaled_averaged_features, features_mean, features_std = scale(averaged_features)

    if apply_pca:
        pca = decomposition.PCA(n_components=10)
        scaled_tpms = pca.fit_transform(scaled_tpms)
        scaled_averaged_features = pca.fit_transform(scaled_averaged_features)

    reduced_rna_matrix = 1 - squareform(pdist(scaled_tpms, metric="cosine"))
    reduced_averaged_features = 1 - squareform(
        pdist(scaled_averaged_features, metric="cosine")
    )
    plt.figure()
    ground_truth_Z = linkage(reduced_rna_matrix)
    dn = dendrogram(ground_truth_Z, labels=cell_columns)
    reduced_rna_matrix = reduced_rna_matrix[dn["leaves"], :][:, dn["leaves"]]
    reduced_averaged_features = reduced_averaged_features[dn["leaves"], :][
        :, dn["leaves"]
    ]

    columns = dn["ivl"]
    plt.figure(figsize=(10, 10))
    plt.imshow(reduced_rna_matrix, cmap="Blues")
    plt.title(f"{label} - RNASeq distance matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print("")

    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.imshow(reduced_averaged_features, cmap="Blues")
    plt.title(f"{label} - DINO similarity matrix")
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    print("")

    for i in range(len(reduced_averaged_features)):
        reduced_averaged_features[i, i] = 0
        reduced_rna_matrix[i, i] = 0

    mantel_test = mantel.test(
        squareform(reduced_averaged_features), squareform(reduced_rna_matrix)
    )
    print(f"{label}, cell line / RNASeq mantel test: {mantel_test}")
    rnaseq_mantel_test = mantel_test[0]

    ca = CCA()
    ca.fit(scaled_tpms, scaled_averaged_features)
    X_c, Y_c = ca.transform(scaled_tpms, scaled_averaged_features)
    cc_res = pd.DataFrame(
        {
            "CCX_1": X_c[:, 0],
            "CCY_1": Y_c[:, 0],
            "CCX_2": X_c[:, 1],
            "CCY_2": Y_c[:, 1],
        }
    )

    cc_distance_matrix = cdist(X_c, Y_c, metric="cosine")

    k = 1
    top_1_accuracy = np.mean(
        (
            np.tile(np.arange(cc_distance_matrix.shape[0])[:, np.newaxis], k)
            == np.argsort(cc_distance_matrix, axis=0)[0:k, :].T
        ).sum(axis=1)
    )
    k = 5
    top_5_accuracy = np.mean(
        (
            np.tile(np.arange(cc_distance_matrix.shape[0])[:, np.newaxis], k)
            == np.argsort(cc_distance_matrix, axis=0)[0:k, :].T
        ).sum(axis=1)
    )
    print(f"{label}, protein CC knn, top-1: {top_1_accuracy}, top-5: {top_5_accuracy}")
    k = 1
    cc_results = (
        np.tile(np.arange(cc_distance_matrix.shape[0])[:, np.newaxis], k)
        == np.argsort(cc_distance_matrix, axis=0)[0:k, :].T
    ).sum(axis=1)
    print(cc_results)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(X_c[:, 0], Y_c[:, 0], color="black")
    plt.xlabel("CC 1 - DINO")
    plt.ylabel("CC 1 - RNASeq")

    cc_res = pd.DataFrame(
        {
            "CCX_1": X_c[:, 0],
            "CCY_1": Y_c[:, 0],
            "CCX_2": X_c[:, 1],
            "CCY_2": Y_c[:, 1],
        }
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    c = [cmap(i / len(cell_columns)) for i in range(len(cell_columns))]

    for i in range(len(X_c)):
        if cc_results[i] == 1:
            linecolor = "black"
        else:
            linecolor = "lightgrey"
        plt.plot(
            [X_c[i, 0], Y_c[i, 0]], [X_c[i, 1], Y_c[i, 1]], c=linecolor, linewidth=3
        )
        plt.scatter(X_c[i, 0], X_c[i, 1], color="red", marker="o", s=25, label="")
        plt.scatter(Y_c[i, 0], Y_c[i, 1], color="blue", marker="o", s=25)

    plt.xlabel("CC 1")
    plt.ylabel("CC 2")

    ax2 = ax.twinx()
    ax2.scatter(np.NaN, np.NaN, color="black", label="RNASeq", marker="*")
    ax2.scatter(np.NaN, np.NaN, color="black", label="DINO features", marker="^")
    ax2.get_yaxis().set_visible(False)
    ax2.legend(
        frameon=False, loc="lower center", title="Dataset", bbox_to_anchor=(0.5, -0.35)
    )

    ax3 = ax.twinx()
    for i in range(len(X_c)):
        plt.scatter(np.NaN, np.NaN, color=c[i], marker="*", s=25, label=cell_columns[i])
    ax3.get_yaxis().set_visible(False)
    ax3.legend(
        frameon=False,
        loc="center right",
        title="Cell types",
        bbox_to_anchor=(1.75, 0.5),
        fontsize=7.5,
        ncol=2,
    )

    return (rnaseq_mantel_test, top_1_accuracy, top_5_accuracy)


def plot_heterogeneous_images(
    config_file,
    merged_df,
    images_sorted,
    scaled_features,
    protein_localizations,
    embedding,
    image_rank,
    cell_preds=None,
    use_well=False,
):
    config = yaml.safe_load(open(config_file, "r"))
    gene2uniprot = pd.read_csv("XML_df/HPA_img_df_new.csv")[
        ["gene", "uniport_id"]
    ].drop_duplicates()
    uniprot2name = pd.read_csv("uniport_interactions.tsv", delimiter="\t")[
        ["Entry", "Gene Names"]
    ].drop_duplicates()
    gene2name = pd.merge(
        uniprot2name, gene2uniprot, right_on="uniport_id", left_on="Entry"
    )[["gene", "Gene Names"]]
    all_df = merged_df.copy()
    gene = merged_df[merged_df.ID == images_sorted[image_rank]].Gene.iloc[0]
    gane_name = gene2name[gene2name.gene == gene]["Gene Names"].iloc[0].split(" ")[0]

    if use_well:
        gene_indices = np.where(
            all_df.ID.apply(lambda x: "_".join(x.split("_")[:2]))
            == "_".join(images_sorted[image_rank].split("_")[:2])
        )
    else:
        gene_indices = np.where(merged_df.ID == images_sorted[image_rank])[0]

    gene_df = merged_df.iloc[gene_indices].reset_index()

    targets = torch.Tensor(protein_localizations)
    color_by = "whole_image"
    if color_by == "whole_image":
        gene_df["well"] = gene_df.ID
        color_by_name = "Image ID"
    elif color_by == "plate":
        gene_df["well"] = gene_df.ID.apply(lambda x: x.split("_")[0])
        color_by_name = "Plate"
    elif color_by == "well":
        gene_df["well"] = gene_df.ID.apply(lambda x: x.split("_")[1])
        color_by_name = "Well"
    dists = pdist(scaled_features[gene_indices], metric="cosine")
    Z = linkage(dists, "ward")

    clustering = DBSCAN(
        eps=scaled_features[gene_indices].std() * 10, min_samples=2
    ).fit(scaled_features[gene_indices])
    clusters = clustering.labels_
    gene_df["cluster"] = clusters
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor="white")
    plt.scatter(np.NaN, np.NaN)

    channels = [1]
    markers = ["*", "^", "o", "|", "v", "s"]
    wells = np.unique(gene_df.well)
    image_IDs = gene_df.ID
    colors = {
        sorted(np.unique(clusters))[i]: "rgbkymrgbkymrgbkym"[i]
        for i in np.unique(clusters)
    }
    # markers = {i : markers[i - 1] for i in np.unique(clusters)}

    cmap_labels = cm.gist_stern
    cmap_image_IDs = cm.nipy_spectral
    image_ID_colors = {
        sorted(np.unique(image_IDs))[i]: cmap_image_IDs(
            ((i + 1) / (len(np.unique(image_IDs)) + 2))
        )
        for i in range(len(np.unique(image_IDs)))
    }
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor="white")
    plt.scatter(
        embedding[:, 0], embedding[:, 1], marker="o", color="grey", alpha=0.5, s=0.01
    )
    for image_ID in sorted(np.unique(image_IDs)):
        indices = np.where(image_IDs == image_ID)[0]
        plt.scatter(
            embedding[gene_indices][indices, 0],
            embedding[gene_indices][indices, 1],
            marker="o",
            color=image_ID_colors[image_ID],
            edgecolors="black",
            label=f"Image ID: {image_ID}",
            s=50,
        )
    plt.legend(loc="center right", bbox_to_anchor=(1.35, 0.5), frameon=False)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.xticks([])
    plt.yticks([])
    # plt.title(
    #     f'UMAP embedding for gene {gane_name}\nrank: {rank + 1}, ground truth: {"heterogeneous" if gene_df.HPA_variable.iloc[0] == True else "unlabeled"}'
    # )

    row_colors1 = gene_df["ID"].map(image_ID_colors)
    row_colors1.name = "Image ID"

    b = ((targets[gene_indices]).numpy()).astype(int)
    c2 = [str(list(i)) for i in b]
    lut1 = dict(
        zip(
            np.unique(c2),
            [
                cmap_labels((i + 1) / (2 + len(np.unique(c2))))
                for i in range(len(np.unique(c2)))
            ],
        )
    )
    series = pd.DataFrame(c2)[0]
    series.name = "Labels"
    row_colors3 = series.map(lut1)
    row_colors = pd.concat([row_colors1, row_colors3], axis=1)

    if cell_preds is not None:
        b = ((cell_preds[gene_indices])).astype(int)
        c1 = [str(list(i)) for i in b]
        series = pd.DataFrame(c1)[0]
        series.name = "Prediction"
        lut1 = dict(
            zip(
                np.unique(c1),
                [
                    cmap_labels(i / len(np.unique(c1)))
                    for i in range(len(np.unique(c1)))
                ],
            )
        )
        row_colors2 = series.map(lut1)
        row_colors = pd.concat([row_colors1, row_colors2, row_colors3], axis=1)

    c = clustermap(
        pd.DataFrame(squareform(dists)),
        figsize=(7.5, 7.5),
        cbar_pos=None,
        xticklabels=[],
        yticklabels=[],
        cmap="Blues_r",
        dendrogram_ratio=[0, 0],
        col_cluster=True,
        row_cluster=True,
        row_colors=row_colors,
        cbar_kws={"size": 20},
        facecolor="white",
    )
    num_to_protein = {
        ind: k for ind, k in enumerate(sorted(protein_to_num_single_cells.keys()))
    }

    for ID in gene_df.ID.unique():
        gene_type = (
            "heterogeneous"
            if merged_df[merged_df.ID == ID].HPA_variable.iloc[0] == True
            else "unlabeled"
        )
        img = io.imread(f"/scr/mdoron/Dino4Cells/data/whole_images/{ID}.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img[:, :, [0, 2, 3]])
        axes[0].axis(False)
        axes[1].imshow(img[:, :, [1]], cmap="Greys_r")
        axes[1].axis(False)
        axes[0].set_title("Non protein channels")
        axes[1].set_title("Protein channel")
        plt.tight_layout()


def plot_heterogeneity(scaled_features, protein_localizations, embedding):
    config_files = [
        "/scr/mdoron/Dino4Cells/configs/config_U-2_OS_0.998.yaml",
        "configs/config_FAIR_varied_masked_with_norm_0.996_5e-4.yaml",
        "/scr/mdoron/Dino4Cells/configs/config_pretrained.yaml",
        "/scr/mdoron/Dino4Cells/configs/config_tta.yaml",
    ]
    labels = [
        "DINO model, trained on U-2 OS data w/ 0.998 center momentum",
        "DINO model, trained on all data w/ 0.9 center momentum",
        "DINO model, trained on ImageNet",
        "2nd place Kaggle competitor",
    ]
    (
        enrichments_conditioned_on_gene,
        enrichments_conditioned_on_image,
        merged_dfs,
        genes_sorted_conditioned_on_gene,
        genes_sorted_conditioned_on_image,
        sorted_metrics_conditioned_on_gene,
        sorted_metrics_conditioned_on_image,
        images_sorted_conditioned_on_image,
        average_precisions_condition_on_gene,
        average_precisions_condition_on_image,
    ) = get_gene_heterogeneity_enrichement(
        config_files, heterogenous_type="HPA_variable"
    )
    plot_gene_heterogeneity_enrichement(
        labels,
        enrichments_conditioned_on_gene,
        enrichments_conditioned_on_image,
        heterogenous_type="HPA_variable",
    )
    plot_heterogeneous_images(
        config_files[0],
        merged_dfs[0],
        images_sorted_conditioned_on_image[0],
        scaled_features,
        protein_localizations,
        embedding,
        image_rank=0,
        cell_preds=None,
        use_well=True,
    )


def get_harmonized_features(df, features):
    ho = hm.run_harmony(features, df, ["cell_type"])
    harmonized_features = ho.Z_corr.T
    (
        umap_reducer,
        features_mean,
        features_std,
        embedding,
        scaled_features,
    ) = get_embeddings(
        torch.Tensor(harmonized_features), torch.Tensor(harmonized_features)
    )
    return embedding, scaled_features
