# DINO4Cells_code
This repo will contain the code for training DINO models and extracting features

# Repo layout

## External data to report
HPA whole images
Cell Painting images
WTC11 single cells
## Internal data to upload
DINO weights for HPA whole image
DINO weights for HPA single cells
DINO weights for Cell painting
DINO weights for WTC11
HPA filtered single cells
HPA normalized images
## Code necessary for training and feature extraction
### Running scripts:
main_dino.py
run_get_features.py
### Utilization files:
utils.py
run_get_kaggle_test_features.py
file_dataset.py
cell_dataset.py
vision_transformer.py
label_dict.py
yaml_tfms.py
### Config files:
Example config file for each dataset

