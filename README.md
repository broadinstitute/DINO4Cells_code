# DINO4Cells_code
This repo will contain the code for training DINO models and extracting features

## Installation

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.py

## Running example

### Train dino
python run_dino.py --config config.yaml --gpus 0,1,2,3
### Extract features
python run_get_features.py --config config.yaml


## External data to report
* HPA whole images
* Cell Painting images
* WTC11 single cells
## Internal data to upload
* DINO weights for HPA whole image
* DINO weights for HPA single cells
* DINO weights for Cell painting
* DINO weights for WTC11
* HPA filtered single cells
* HPA normalized images
