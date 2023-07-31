
## Example training

For this example, we assume we have a dataset of 10,000 FOV images of cells. We wish to train an unbiased feature extractror to explore the structure of the data. To do this, we should first prepare a configuration file that will contain the parapeters DINO will use to train:

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
     epochs: 10
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
