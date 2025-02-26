
import shutil
import glob
import tqdm
import os

data_dirs_dino = [
    "data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small/final-datasets/dino/train",
    "data/CheXpert/final-datasets/dino/train",
    "data/PadChest/final-datasets/dino/train",
    "data/ChestXRay-14/CXR8/final-datasets/dino/train",
    "data/NLM-CXR/final-datasets/dino/train"
]

data_dirs_clip = [
    "data/MIMIC-CXR/mimic-cxr-jpg-2.0.0-small/final-datasets/clip/train",
    "data/CheXpert/final-datasets/clip/train",
    "data/PadChest/final-datasets/clip/train",
    "data/ChestXRay-14/CXR8/final-datasets/clip/train",
    "data/NLM-CXR/final-datasets/clip/train"
]

print("Creating output directories data/dino_train_images_flat and data/clip_train_images_flat")
os.makedirs("data/dino_train_images_flat", exist_ok=True)
os.makedirs("data/clip_train_images_flat", exist_ok=True)

errors = []
def combine_images(data_dir, output_dir):
    print(f"Combining images from {data_dir} to {output_dir}")
    images = glob.glob(data_dir + "/*")
    print(f"Found {len(images)} files in {data_dir}")
    for image in tqdm.tqdm(images):
        try:
            shutil.copy(image, output_dir)
        except Exception as e:
            print(f"Error copying {image} to {output_dir}: {e}")
            errors.append(image)
    if len(errors) > 0:
        print(f"Found {len(errors)} errors")
    else:
        print("No errors found")


for data_dir in data_dirs_dino:
    combine_images(data_dir, "data/dino_train_images_flat")

for data_dir in data_dirs_clip:
    combine_images(data_dir, "data/clip_train_images_flat")
