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
