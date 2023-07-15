# DINO4Cells_code
This repo will contain the code for training DINO models and extracting features, as described in the paper [Unbiased single-cell morphology with self-supervised vision transformers
](https://www.biorxiv.org/content/10.1101/2023.06.16.545359v1).

For the code to reproduce the results of the paper, go to https://github.com/broadinstitute/Dino4Cells_analysis.

## Data

For the WTC11 data, access [https://zenodo.org/record/8061424](https://zenodo.org/record/8061424)
For the Cell Painting data, access [https://zenodo.org/record/8061428](https://zenodo.org/record/8061428)
For the HPA single cell data, access [https://zenodo.org/record/8061426](https://zenodo.org/record/8061426)
For the HPA FOV data, access [https://zenodo.org/record/8061392](https://zenodo.org/record/8061392)

## Installation

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.py

Typical installation time: 10 minutes

## Running example

### Train dino
python run_dino.py --config config.yaml --gpus 0,1,2,3
Typical running time: 1 day

### Extract features
python run_get_features.py --config config.yaml

### Train classifier
CUDA_VISIBLE_DEVICES=0,1,2,3 python  -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config.yaml --epochs 100 --balance True --num_classes 35 --train_cell_type True --train_protein False --master_port = 1234
