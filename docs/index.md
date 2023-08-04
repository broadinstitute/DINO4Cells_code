# Dino4cells handbook

Dino4cells is a self-supervised method to extract phenotypic information of single cells from their morphology by training unbiased representation learning models on their microscopy images. This guide describes how to install dino4cells, and how to run it on an example small dataset. After completing this guidebook, you should have a good understanding of how to use dino4cells in your own research. 

Dino4cells is an extention of [DINO](https://github.com/facebookresearch/dino), or self-DIstilation with NO labels, published by Meta-AI. A demonstration of its abilities is presented in the paper [Unbiased single-cell morphology with self-supervised vision transformers](https://www.biorxiv.org/content/10.1101/2023.06.16.545359v1).


## Installation

To install the code, first please clone the [DINO4cells_code git repo](https://github.com/broadinstitute/Dino4Cells_code).

Next, install the required dependencies by inputing

`pip install -r requirements.py`
and
`pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`

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


## Example training

For this example, we assume we have a dataset of 15188 images of single cells, that can be downloaded from [here](https://zenodo.org/record/8198252). We wish to train an unbiased feature extractror to explore the structure of the data. To do this, we should first prepare a configuration file that will contain the parapeters DINO will use to train:

Inside `config.yaml`:

```
   ...
   model:
     model_type: DINO
     arch: vit_tiny
     root: /home/michaeldoron/dino_example/single_cells_data/
     data_path: /home/michaeldoron/dino_example/metadata.csv
     output_dir: /home/michaeldoron/dino_example/output/
     datatype: HPA
     image_mode: normalized_4_channels
     batch_size_per_gpu: 24
     num_channels: 4
     patch_size: 16
     epochs: 100
     momentum_teacher: 0.996
     center_momentum: 0.9
     lr: 0.0005
     local_crops_scale: '0.2 0.5'
    ... 
```

These are the parameters we need to set for DINO to train.

`arch` can be `vit_tiny`, `vit_small` or `vit_base`, giving us options for a larger feature vector with the price of more memory and computation.

`root` points to the root directory where the images are stored.

`data_path` points to the csv file containing the metadata.

`output_dir` points to the location where the DINO model will be stored.

`image_mode` described the type of images the data loader should expect to find. In this case, we use normalized 4 channels, as our data will contain 4 channel images. You can create new image modes by changing the code in `data_utils/file_dataset.py`.

`batch_size_per_gpu` determines the size of the batch per each GPU. 

`num_channels` determines the number of channels the DINO model should expect.

`patch_size` determines the size of the vision transformer patch. Usual values are 8 or 16 pixels.

`epochs` determines the number of epochs the DINO model shall train over the entire data.

`momentum_teacher` determines the momentum used to transfer the weights from the student network to the teacher network.

`center_momentum` determines the momentum used to center the teacher.

`lr` determines the learning rate used by DINO.

`local_crops_scale` determines the size of the crops used by DINO.


After the determine these parameters, we can now decide on the augmentations DINO will use.

As written in the paper, DINO relies on a selection of augmentations that transform the images in ways DINO learns to be invariant to in its feature extraction. Thus, to train DINO well, one should choose augmentations that do not alter the important information found inside the images. These augmentations can be altered in these sections of the DINO `config.yaml` file: `flip_and_color_jitter_transforms`, `normalization`, `global_transfo1`, `global_aug1`, `testing_transfo`, `global_transfo2`, `local_transfo`, `local_aug`. 

Each section is active at the different time, in the global views, local views, or normalization. Inside each section there is a list of augmentations, each with a boolean flag signifying whether it is active or not in the training process, as well as possible additional augmentation parameters. 

For example:

```
 global_transfo1:
   Warp_cell:
    - True
    - # no params
   Single_cell_centered:
    - False
    - # no params    
   remove_channel:
    - True
    - {p: 0.2}
```

means that the global views has the Warp_cell augmentation active, the single_cell_centering augmentation not active, and the remove_channel augmentation active, with a probability of 20% activation.

After the config file is set, we can train our DINO model on the data by running `python run_dino.py --config config.yaml --gpus 0,1`, where the `--gpus` argument determines which GPUs are used in training. 

On a single 3090 GPU, training DINO on 15k images should take about 3.5 hours.


## Example extraction

After we trained DINO of the images, we can now use it to extract features.

For this, we will again look at the `config.yaml` file, this time at the `embeddings` section.

Inside `config.yaml`:

```
...
 embedding:
   pretrained_weights: /scr/mdoron/DINO4Cells_code/output/checkpoint.pth
   output_path: /scr/mdoron/DINO4Cells_code/output/features.pth
   df_path: /scr/mdoron/DINO4Cells_code/cellpainting_data/sc-metadata.csv
   image_size: 224
   num_workers: 0
   embedding_has_labels: True
   target_labels: False
...
```

These are the parameters we need to set for DINO to extract features.

`pretrained_weights` points to the checkpoint of the trained DINO model

`output_path` points to the location where the DINO features will be saved

`df_path` points to the csv file containing the metadata of the data to be extracted

`image_size` determines the size of the images DINO expects to find

`num_workers` determines how many workers will be used in extracting the features

`embedding_has_labels` determines whether the metadata has labels to be saved along with the features

`target_labels` False


Finally, run this command to extract the features: 

`python run_get_features.py --config config.yaml`


