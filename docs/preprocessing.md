## Pretraining datasets:

### MIMIC-CXR:
1. Parse "findings" and "impression" from the reports using `mimic_section_parser.py`
2. Create the dino, clip folders and the csv file with labels and captions: `python -i data_processing/dataset_builders/mimic.py --base_path data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small --output_image_dir data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small/final-datasets`

### CheXpert:
Similarly:
`python -i data_processing/dataset_builders/chexpert.py --base_path data/CheXpert --output_image_dir data/CheXpert/final-datasets/ --existing_jpg_dir data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small/final-datasets/dino/train `

### PadChest:
1. Translate all reports using `padchest_openai.py`. This will dump a csv file called `PadChest-Translated.csv`.
   - `python -i padchest_openai.py --input_file ../data/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv --output_file ../data/PadChest/PadChest-Translated.csv --cache_file ../data/PadChest/translations.csv;`
2. Create the dino, clip folders using `python -i padchest.py --base_path ../../data/PadChest --output_image_dir ../../data/PadChest/final-datasets;`


### ChestX-ray14
`python -i nih.py --base_path ../../data/ChestXRay-14/CXR8 --output_path ../../data/ChestXRay-14/CXR8/final-datasets --label-to-text-file label-to-text.json`


### NLM-CXR/OpenI
- `python -i nlmcxr.py --base_path ../../data/NLM-CXR --output_path ../../data/NLM-CXR/final-datasets  `



### A script to sample 500 images from each dataset and copy them into a single folder to inspect visually
```bash
for dir in data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small/final-datasets/dino/train data/CheXpert/final-datasets/dino/train data/PadChest/final-datasets/dino/train data/ChestXRay-14/CXR8/final-datasets/dino/train data/NLM-CXR/final-datasets/dino/train; do
    dataset_name=$(echo "$dir" | cut -d'/' -f2)
    mkdir -p "data/random-sample/$dataset_name"
    find "$dir" -type f | wc -l
done
;
```

- Combine all the clip/train folders into one using `data_processing.combine_images.py`
- Tar: 
  - `tar -I zstd -cvf data/dino_train_images_flat.tar.zst data/dino_train_images_flat`
  - `tar -I zstd -cvf data/clip_train_images_flat.tar.zst data/clip_train_images_flat`
- Untar: 
  - `tar -I zstd -xvf data/dino_train_images_flat.tar.zst -C /path/to/destination`
  - `tar -I zstd -xvf data/clip_train_images_flat.tar.zst -C /path/to/destination`

- Convert the clip folder into several tar files (as required by open_clip) using `clip-tar-prepare.py`




### VinDr-CXR
Convert the dicom files into jpgs and resize them into 224x224
`python vindr-cxr.py --input ../../data/VinDr-CXR/vinbigdata-chest-xray-abnormalities-detection.zip --output ../../data/VinDr-CXR/images/ --num-workers 12`

### VinDr-PCXR
similar as above, but with `python vindr-pcxr.py`

### Eval datasets
Follow the notebook `data_processing/explore_eval_data.ipynb`


