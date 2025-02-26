import pandas as pd
import argparse
import os
from tqdm import tqdm
from PIL import Image
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--output_image_dir', type=str, required=True)
    return parser.parse_args()

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def resize_image(input_path, output_path, size=(224, 224)):
    with Image.open(input_path) as img:
        # Check if the image is in 'I;16' mode and convert it to 8-bit grayscale
        if img.mode == 'I;16':
            img = img.point(lambda i: i * (1./256)).convert('L')
        img = img.resize(size, Image.BICUBIC)
        img = img.convert('RGB')  # Ensure the image is in RGB mode
        img.save(output_path, 'JPEG')

def process_images_and_captions(df, base_path, output_image_dir):
    create_folder(os.path.join(output_image_dir, 'dino'))
    create_folder(os.path.join(output_image_dir, 'dino', 'train'))
    create_folder(os.path.join(output_image_dir, 'clip'))
    create_folder(os.path.join(output_image_dir, 'clip', 'train'))

    print("Processing images for DINO...")
    valid_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(base_path, "images/images-224", row['ImageID'])
        if not os.path.exists(img_path):
            continue
        
        new_filename = os.path.splitext(row['ImageID'])[0] + '.jpg'
        dino_jpg_path = os.path.join(output_image_dir, 'dino', 'train', new_filename)
        
        resize_image(img_path, dino_jpg_path)
        
        row['new_path'] = os.path.join('dino', 'train', new_filename)
        valid_rows.append(row)

    df_valid = pd.DataFrame(valid_rows)

    print("Copying DINO to CLIP...")
    shutil.copytree(os.path.join(output_image_dir, 'dino', 'train'), 
                    os.path.join(output_image_dir, 'clip', 'train'), 
                    dirs_exist_ok=True)

    print("Writing captions for CLIP...")
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid)):
        clip_txt_path = os.path.join(output_image_dir, 'clip', 'train', f"{os.path.splitext(row['ImageID'])[0]}.txt")
        with open(clip_txt_path, 'w') as f:
            f.write(row['TranslatedReport'])

    return df_valid

args = parse_args()

df = pd.read_csv(os.path.join(args.base_path, "PadChest-Translated.csv"))
df = df[df['TranslatedReport'].notna()]
df = df[df['TranslatedReport'] != '']

df = process_images_and_captions(df, args.base_path, args.output_image_dir)

df.to_csv(os.path.join(args.output_image_dir, 'padchest_processed.csv'), index=False)


