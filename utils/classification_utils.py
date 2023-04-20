import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn.functional import sigmoid
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from random import choices
import sys

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
    if args.schedule == "Cosine":
        scheduler = CosineAnnealingLR(optim, int(total_steps))
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
    elif args.classifier_type == "residual_add_clf":
        classifier = residual_add_clf(
            embed_dim,
            args.num_classes,
            n_layers=int(args.n_layers),
            n_units=int(args.n_units),
            norm_layer=args.norm_layer if "norm_layer" in args else None,
            skip=args.skip,
        )
    return classifier

