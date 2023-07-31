```{tableofcontents}
```

# Dino4cells handbook

Dino4cells is a self-supervised method to extract phenotypic information of single cells from their morphology by training unbiased representation learning models on their microscopy images. This guide describes how to install dino4cells, and how to run it on an example small dataset. After completing this guidebook, you should have a good understanding of how to use dino4cells in your own research. 

Dino4cells is an extention of [DINO](https://github.com/facebookresearch/dino), or self-DIstilation with NO labels, published by Meta-AI. A demonstration of its abilities is presented in the paper [Unbiased single-cell morphology with self-supervised vision transformers](https://www.biorxiv.org/content/10.1101/2023.06.16.545359v1).


## Installation

To install the code, first please clone the [DINO4cells_code git repo](https://github.com/broadinstitute/Dino4Cells_code).

Next, install the required dependencies by inputing

`pip install -r requirements.py`
and
`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

This should make you set up for running DINO4cells.


## Usage

### Train dino
`python run_dino.py --config config.yaml --gpus 0,1,2,3` 

Typical running time: 1 day

### Extract features
After the model is trained, you can extract features from the microscopy images by running

`python run_get_features.py --config config.yaml`

### Train classifier
Next, if you want to use the features for, e.g., predicting some quantity of interest, you can run

`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_end_to_end.py --config config.yaml --epochs 100 --balance True --num_classes 35 --train_cell_type True --train_protein False --master_port = 1234`

