from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch
import glob


# https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L41C2-L50C52

class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to 
    one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)
    
# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CXR_MEAN = (0.4958348274, 0.4958348274, 0.4958348274)
CXR_STD = (0.2771022319, 0.2771022319, 0.2771022319)

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        return img

class GrayscaleToRGB:
    "Custom transform to convert 1-channel image to 3-channel image."
    def __call__(self, img):
        if img.size(0) == 1:  # Check if the image has 1 channel
            img = img.repeat(3, 1, 1)  # Repeat the channel 3 times
        return img

def make_classification_train_transform(*,
                                        mean=CXR_MEAN,
                                        std=CXR_STD,
                                        ) -> transforms.Compose:
    """This roughly matches torchvision's preset for classification training
    https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44"""
    transforms_list = [
        MaybeToTensor(),
        GrayscaleToRGB(),
        Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def make_classification_eval_transform(*,
                                       mean = CXR_MEAN,
                                       std = CXR_STD,
                                       ):
    """This matches (roughly) torchvision's preset for classification evaluation
    https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69"""
    transforms_list = [
        MaybeToTensor(),
        GrayscaleToRGB(),
        Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def check_sampling_criteria_multilabel(data, min_samples_per_label):
    for label in data.columns.tolist():
        if data[label].sum() < min_samples_per_label:
                return False
    return True

def check_sampling_criteria_singlelabel(data, min_samples_per_label):
    label_counts = data['label'].value_counts().to_dict()
    for label, count in label_counts.items():
        if count < min_samples_per_label:
            return False
    return True

class MIMICCXRDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.data = pd.read_csv(args['label_file'], sep='\t')
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.split = split
        self.data = self.data[self.data['split'] == split]
        self.label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                              'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
                              'Lung Opacity', 'No Finding', 'Pleural Effusion', 
                              'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        self.data[self.label_columns] = self.data[self.label_columns].replace(-1, 1)
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
        
        self.metadata_columns = ['age', 'sex']  # Add metadata columns
        
        # Fill NaN values for metadata
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('unknown')
        
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*/*/*/*.jpg")
        # Drop rows where the image is not found
        for idx, row in self.data.iterrows():
            path = row['path']
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # dicom_id = row['dicom_id']
        # img_path = os.path.join(self.image_dir, f"{self.split}", f"{dicom_id}.jpg")
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")        

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        
        # Add metadata to output
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'path': img_path  # Including path can be useful for analysis
        }
        
        return image, labels, metadata
    
class MIMICCXRLTDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_columns = [
            'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture', 'Hernia',
            'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'No Finding', 'Nodule',
            'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum',
            'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema',
            'Support Devices', 'Tortuous Aorta'
        ]
        
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*/*/*/*.jpg")
        # Drop rows where the image is not found
        for idx, row in self.data.iterrows():
            path = row['path']
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
    
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        return image, labels

class ChexpertDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_columns = ['Enlarged Cardiomediastinum', 'Cardiomegaly',
                              'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                              'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                              'Fracture', 'Support Devices', 'No Finding']
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        self.data[self.label_columns] = self.data[self.label_columns].replace(-1, 1)
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
                
        self.metadata_columns = ['age', 'sex', 'race']
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('Unknown')
        
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*/*/*/*.jpg")
        # Drop rows where the image is not found
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path_to_image'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path_to_image'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'race': row['race'],
            'path': img_path
        }
        
        return image, labels, metadata
    
class NIHLTDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'])
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                            'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax',
                            'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema',
                            'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding']
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        # Fill NaN values for metadata
        self.metadata_columns = ['age', 'sex']
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('Unknown')
        
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'path': img_path
        }
        
        return image, labels, metadata

class NIHDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'])
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_columns = ['Effusion', 'Cardiomegaly', 'Hernia', 'Infiltration', 
                            'Mass', 'Pneumonia', 'Pleural_Thickening', 'No Finding',
                            'Edema', 'Nodule', 'Pneumothorax', 'Consolidation',
                            'Fibrosis', 'Atelectasis', 'Emphysema']
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        # Fill NaN values for metadata
        self.metadata_columns = ['age', 'sex']
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('Unknown')
        
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'path': img_path
        }
        
        return image, labels, metadata

class COVIDDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'])
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_column = 'label'
        
        self.data[self.label_column] = self.data[self.label_column].fillna(0)
        
        print("Distribution of samples per class:")
        class_counts = self.data[self.label_column].value_counts()
        for label, count in class_counts.items():
            print(f"Class {label}: {count/len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_singlelabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return image, label

class ChexchonetDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'])
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_column = 'label'
        
        self.data[self.label_column] = self.data[self.label_column].fillna(0)
        
        print("Distribution of samples per class:")
        class_counts = self.data[self.label_column].value_counts()
        for label, count in class_counts.items():
            print(f"Class {label}: {count/len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_singlelabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return image, label
    
class TBX11KDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_column = 'label'
        
        self.data[self.label_column] = self.data[self.label_column].fillna(0)
        
        if args['sampling_ratio']:
            print(f"Sampling {args['sampling_ratio']*100}% of the data")
            while True:
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_singlelabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
        
        print("Distribution of samples per class:")
        class_counts = self.data[self.label_column].value_counts()
        for label, count in class_counts.items():
            print(f"Class {label}: {count/len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*/*.jpg")
        all_img_paths += glob.glob(f"{os.path.join(self.image_dir)}/*/*/*.jpg")
        all_img_paths += glob.glob(f"{os.path.join(self.image_dir)}/*/*/*/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return image, label
    
class RSNAPneumoniaDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_column = 'label'
        
        self.data[self.label_column] = self.data[self.label_column].fillna(0)
        
        print("Distribution of samples per class:")
        class_counts = self.data[self.label_column].value_counts()
        for label, count in class_counts.items():
            print(f"Class {label}: {count/len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_singlelabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return image, label
    
class SIIMPneumothoraxDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_column = 'label'
        
        self.data[self.label_column] = self.data[self.label_column].fillna(0)
        
        print("Distribution of samples per class:")
        class_counts = self.data[self.label_column].value_counts()
        for label, count in class_counts.items():
            print(f"Class {label}: {count/len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_singlelabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row[self.label_column], dtype=torch.long)
        
        return image, label
    
class VindrCXRDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_columns = ['Aortic enlargement', 'Atelectasis', 'Calcification',
                      'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema',
                      'Emphysema', 'Enlarged PA', 'ILD', 'Infiltration', 'Lung Opacity',
                      'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
                      'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                      'Pulmonary fibrosis', 'Rib fracture', 'COPD', 'Lung tumor', 'Pneumonia',
                      'Tuberculosis', 'No finding']
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)

        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
        
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        
        return image, labels
    
class VindrPCXRDataset(Dataset):
    def __init__(self, args, split):
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        if args['seed']:
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
        self.transform = None if args['no_transform'] else (
            make_classification_train_transform() if split == 'train' 
            else make_classification_eval_transform()
        )
        self.label_columns = ['No finding', 'Bronchitis', 'Brocho-pneumonia', 'Bronchiolitis',
                            'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia',
                            'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema', 'CPAM',
                            'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor']
        
        self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        
        if args['drop_columns']:
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        print("Ratio of positive samples per class:")
        pos_counts = self.data[self.label_columns].sum()
        for col in self.label_columns:
            print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        print()
        
        dropped_count = 0
        all_img_paths = glob.glob(f"{os.path.join(self.image_dir)}/*.jpg")
        for idx, row in self.data.iterrows():
            path = os.path.join(self.image_dir, row['path'])
            if path not in all_img_paths:
                self.data = self.data.drop(idx)
                dropped_count += 1
        print(f"Dropped {dropped_count} rows where the image was not found")
        
        if args['sampling_ratio']:
            while True:
                print(f"Sampling {args['sampling_ratio']*100}% of the data")
                sample = self.data.sample(frac=args['sampling_ratio'], random_state=args['seed']).reset_index(drop=True)
                if check_sampling_criteria_multilabel(sample, args['min_samples_per_label']):
                    self.data = sample
                    break
                else:
                    print(f"Sampling criteria not met. Retrying.")
                    args['seed'] += 1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['path'])
        img_path = os.path.join(self.image_dir, img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(float))
        
        return image, labels
    
def get_dataset(training_args):
    dataset_name = training_args['finetune_dataset']
    label_file = training_args['label_file']
    image_dir = training_args['image_dir']
    seed = training_args['seed'] if 'seed' in training_args else None
    sampling_ratio = training_args['sampling_ratio'] if 'sampling_ratio' in training_args else None
    min_samples_per_label = training_args['min_samples_per_label'] if 'min_samples_per_label' in training_args else None
    drop_columns = training_args['drop_columns'] if 'drop_columns' in training_args else None
    no_transform = training_args['no_transform'] if 'no_transform' in training_args else False
    
    if drop_columns:
        drop_columns = drop_columns.split(";")
        
    args = {
        'dataset_name': dataset_name,
        'label_file': label_file,
        'image_dir': image_dir,
        'seed': seed,
        'sampling_ratio': sampling_ratio,
        'min_samples_per_label': min_samples_per_label,
        'drop_columns': drop_columns,
        'no_transform': no_transform
    }
    
    if dataset_name == "mimic":
        train_dataset = MIMICCXRDataset(args, split='train')
        val_dataset = MIMICCXRDataset(args, split='validate')
        test_dataset = MIMICCXRDataset(args, split='test')
    elif dataset_name == "mimic_lt":
        train_dataset = MIMICCXRLTDataset(args, split='train')
        val_dataset = MIMICCXRLTDataset(args, split='validate')
        test_dataset = MIMICCXRLTDataset(args, split='test')
    elif dataset_name == "chexpert":
        train_dataset = ChexpertDataset(args, split='train')
        val_dataset = ChexpertDataset(args, split='valid')
        test_dataset = ChexpertDataset(args, split='test')
    elif dataset_name == "nih":
        train_dataset = NIHDataset(args, split='train')
        val_dataset = NIHDataset(args, split='valid')
        test_dataset = NIHDataset(args, split='test')
    elif dataset_name == "nih_lt":
        train_dataset = NIHLTDataset(args, split='train')
        val_dataset = NIHLTDataset(args, split='valid')
        test_dataset = NIHLTDataset(args, split='test')
    elif dataset_name == "covid":
        train_dataset = COVIDDataset(args, split='train')
        val_dataset = COVIDDataset(args, split='valid')
        test_dataset = COVIDDataset(args, split='test')
    elif dataset_name == "chexchonet":
        train_dataset = ChexchonetDataset(args, split='train')
        val_dataset = None
        test_dataset = ChexchonetDataset(args, split='test')
    elif dataset_name == "tbx11k":
        train_dataset = TBX11KDataset(args, split='train')
        val_dataset = None
        test_dataset = TBX11KDataset(args, split='test')
    elif dataset_name == "rsna":
        train_dataset = RSNAPneumoniaDataset(args, split='train')
        val_dataset = None
        test_dataset = RSNAPneumoniaDataset(args, split='test')
    elif dataset_name == "siim":
        train_dataset = SIIMPneumothoraxDataset(args, split='train')
        val_dataset = None
        test_dataset = SIIMPneumothoraxDataset(args, split='test')
    elif dataset_name == "vindr-cxr":
        train_dataset = VindrCXRDataset(args, split='train')
        val_dataset = None
        test_dataset = VindrCXRDataset(args, split='test')
    elif dataset_name == "vindr-pcxr":
        train_dataset = VindrPCXRDataset(args, split='train')
        val_dataset = None
        test_dataset = VindrPCXRDataset(args, split='test')
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
        
    return train_dataset, val_dataset, test_dataset
        