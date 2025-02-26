from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch
import glob
from typing import List, Optional, Dict, Any, Tuple

class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)

# Constants for normalization
CXR_MEAN = (0.4958348274, 0.4958348274, 0.4958348274)
CXR_STD = (0.2771022319, 0.2771022319, 0.2771022319)

class Normalize:
    """Normalize with given mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, img):
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        return img

class GrayscaleToRGB:
    """Convert 1-channel image to 3-channel image."""
    def __call__(self, img):
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

def make_classification_train_transform(*, mean=CXR_MEAN, std=CXR_STD) -> transforms.Compose:
    """Create transform pipeline for training."""
    transforms_list = [
        MaybeToTensor(),
        GrayscaleToRGB(),
        Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def make_classification_eval_transform(*, mean=CXR_MEAN, std=CXR_STD):
    """Create transform pipeline for evaluation."""
    transforms_list = [
        MaybeToTensor(),
        GrayscaleToRGB(),
        Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def check_sampling_criteria_multilabel(data, min_samples_per_label, label_columns):
    for label in data[label_columns].columns.tolist():
        if data[label].sum() < min_samples_per_label:
            return False
    return True

def check_sampling_criteria_singlelabel(data, min_samples_per_label, label_columns="label"):
    label_counts = data["labels"].value_counts().to_dict()
    for label, count in label_counts.items():
        if count < min_samples_per_label:
            return False
    return True

class BaseXRayDataset(Dataset):
    """Base class for X-ray datasets with common functionality."""
    
    def __init__(self, 
                 args: Dict[str, Any], 
                 split: str,
                 label_columns: List[str],
                 multilabel: bool = True,
                 replace_png_with_jpg: bool = False):
        """
        Args:
            args: Dictionary containing dataset configuration
            split: Data split ('train', 'valid', 'test')
            label_columns: List of column names containing labels
            multilabel: Whether dataset is multilabel or single-label classification
        """
        self.image_dir = args['image_dir']
        self.data = pd.read_csv(args['label_file'], sep='\t')
        self.data = self.data[self.data['split'] == split]
        self.label_columns = label_columns
        self.multilabel = multilabel
        
        # Initialize transforms
        # self.transform = None if args.get('no_transform') else (
        #     make_classification_train_transform() if split == 'train' 
        #     else make_classification_eval_transform()
        # )
        if args.get('no_transform'):
            # Even if no extra transform is specified, convert images to tensor
            self.transform = transforms.ToTensor()
        else:
            self.transform = make_classification_train_transform() if split == 'train' \
                             else make_classification_eval_transform()
        
        # Apply seed if specified
        if args.get('seed'):
            print(f"Sampling seed {args['seed']} of the data")
            self.data = self.data.sample(frac=1, random_state=args['seed']).reset_index(drop=True)
            
        # Handle NaN labels
        if multilabel:
            self.data[self.label_columns] = self.data[self.label_columns].fillna(0)
        else:
            self.data[self.label_columns[0]] = self.data[self.label_columns[0]].fillna(0)
            
        self.data[self.label_columns] = self.data[self.label_columns].replace(-1, 1)
            
        # Drop specified columns
        if args.get('drop_columns'):
            for column in args['drop_columns']:
                print(f"Dropping column: {column}")
                self.data = self.data[self.data[column] == 0]
            
        if replace_png_with_jpg: # This exists because of a preprocessing bug in the NIH-LT dataset where paths are in .png format
            self.data['path'] = self.data['path'].str.replace('.png', '.jpg')
                
        # Print class distribution
        self._print_class_distribution()
        
        # Handle missing images
        self._handle_missing_images()
        
        # Apply sampling if specified
        if split=="train" and args.get('sampling_ratio'):
            self._apply_sampling(args)

    def _print_class_distribution(self):
        """Print distribution of classes/labels."""
        print("\nClass distribution:")
        if self.multilabel:
            pos_counts = self.data[self.label_columns].sum()
            for col in self.label_columns:
                print(f"{col}: {pos_counts[col] / len(self.data):.4f}")
        else:
            class_counts = self.data[self.label_columns[0]].value_counts()
            for label, count in class_counts.items():
                print(f"Class {label}: {count/len(self.data):.4f}")
        print()

    def _get_glob_pattern(self) -> str:
        """Get dataset-specific glob pattern for image paths."""
        return f"{os.path.join(self.image_dir)}/**/*.jpg"

    def _handle_missing_images(self):
        """Remove entries where image files are missing."""
        dropped_count = 0
        glob_pattern = self._get_glob_pattern()
        
        # Handle special case for TBX11K dataset which returns a set
        if isinstance(glob_pattern, set):
            all_img_paths = glob_pattern
        else:
            all_img_paths = set(glob.glob(glob_pattern, recursive=True))
        
        drop_indices = []
        for idx, row in self.data.iterrows():
            path = self._get_image_path(row)
            if path not in all_img_paths:
                drop_indices.append(idx)
                dropped_count += 1
                
        self.data = self.data.drop(drop_indices)
        print(f"Dropped {dropped_count} rows where images were not found")

    def _apply_sampling(self, args: Dict[str, Any]):
        """Apply sampling strategy if specified."""
        while True:
            print(f"Sampling {args['sampling_ratio']*100}% of the data")
            sample = self.data.sample(frac=args['sampling_ratio'], 
                                    random_state=args['seed']).reset_index(drop=True)
            
            check_func = (check_sampling_criteria_multilabel if self.multilabel 
                         else check_sampling_criteria_singlelabel)
            
            if check_func(sample, args['min_samples_per_label'], self.label_columns):
                self.data = sample
                break
            else:
                print("Sampling criteria not met. Retrying.")
                args['seed'] += 1

    def _get_image_path(self, row: pd.Series) -> str:
        """Get full image path from row data."""
        img_path = str(row.get('path_to_image', row.get('path')))
        return os.path.join(self.image_dir, img_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img_path = self._get_image_path(row)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        if self.multilabel:
            labels = torch.tensor(row[self.label_columns].values.astype(float))
        else:
            labels = torch.tensor(row[self.label_columns[0]], dtype=torch.long)
            
        return image, labels

class MIMICCXRDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        super().__init__(args, split, label_columns, multilabel=True)        
        
        self.metadata_columns = ['age', 'sex']
        # Fill NaN values for metadata
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('unknown')

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*/*/*/*/*.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image, labels = super().__getitem__(idx)
        row = self.data.iloc[idx]
        
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'path': self._get_image_path(row)
        }
        
        return image, labels, metadata

class MIMICCXRLTDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 
            'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture', 
            'Hernia', 'Infiltration', 'Lung Lesion', 'Lung Opacity', 'Mass', 'No Finding', 
            'Nodule', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 
            'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 
            'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta'
        ]
        super().__init__(args, split, label_columns, multilabel=True)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*/*/*/*/*.jpg"

class ChexpertDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 
            'Support Devices', 'No Finding'
        ]
        super().__init__(args, split, label_columns, multilabel=True)
        
        # CheXpert-specific preprocessing
        self.data[self.label_columns] = self.data[self.label_columns].replace(-1, 1)
        self.metadata_columns = ['age', 'sex', 'race']
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('Unknown')

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*/*/*/*.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image, labels = super().__getitem__(idx)
        row = self.data.iloc[idx]
        
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'race': row['race'],
            'path': self._get_image_path(row)
        }
        
        return image, labels, metadata

class NIHDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'Effusion', 'Cardiomegaly', 'Hernia', 'Infiltration', 'Mass', 
            'Pneumonia', 'Pleural_Thickening', 'No Finding', 'Edema', 'Nodule', 
            'Pneumothorax', 'Consolidation', 'Fibrosis', 'Atelectasis', 'Emphysema'
        ]
        super().__init__(args, split, label_columns, multilabel=True)
        
        self.metadata_columns = ['age', 'sex']
        self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('Unknown')
        
    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image, labels = super().__getitem__(idx)
        row = self.data.iloc[idx]
        
        metadata = {
            'age': row['age'],
            'sex': row['sex'],
            'path': self._get_image_path(row)
        }
        
        return image, labels, metadata

class NIHLTDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
            'Pleural Thickening', 'Pneumonia', 'Pneumothorax', 'Pneumoperitoneum',
            'Pneumomediastinum', 'Subcutaneous Emphysema', 'Tortuous Aorta',
            'Calcification of the Aorta', 'No Finding'
        ]
        super().__init__(args, split, label_columns, multilabel=True, replace_png_with_jpg=True) 
        # Replace png with jpg is true because of a preprocessing bug in the NIH-LT dataset where paths are in .png format
        
        # self.metadata_columns = ['age', 'sex']
        # self.data[self.metadata_columns] = self.data[self.metadata_columns].fillna('Unknown')
        
    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        image, labels = super().__getitem__(idx)
        # row = self.data.iloc[idx]
        
        # metadata = {
        #     'age': row['age'],
        #     'sex': row['sex'],
        #     'path': self._get_image_path(row)
        # }
        
        return image, labels

    
class VindrCXRDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
            'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
            'ILD', 'Infiltration', 'Lung Opacity', 'Lung cavity', 'Lung cyst',
            'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening',
            'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture', 'COPD', 'Lung tumor',
            'Pneumonia', 'Tuberculosis', 'No finding'
        ]
        super().__init__(args, split, label_columns, multilabel=True)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"

class VindrPCXRDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = [
            'No finding', 'Bronchitis', 'Brocho-pneumonia', 'Bronchiolitis',
            'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia',
            'Tuberculosis', 'Congenital emphysema', 'CPAM', 'Hyaline membrane disease',
            'Mediastinal tumor', 'Lung tumor'
        ]
        super().__init__(args, split, label_columns, multilabel=True)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"


class COVIDDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = ['label']
        super().__init__(args, split, label_columns, multilabel=False)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"

class ChexchonetDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = ['label']
        super().__init__(args, split, label_columns, multilabel=False)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"

class TBX11KDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = ['label']
        super().__init__(args, split, label_columns, multilabel=False)

    def _get_glob_pattern(self) -> str:
        patterns = [
            f"{os.path.join(self.image_dir)}/*.jpg",
            f"{os.path.join(self.image_dir)}/*/*.jpg",
            f"{os.path.join(self.image_dir)}/*/*/*.jpg",
            f"{os.path.join(self.image_dir)}/*/*/*/*.jpg"
        ]
        all_paths = []
        for pattern in patterns:
            all_paths.extend(glob.glob(pattern))
        return set(all_paths)

class RSNAPneumoniaDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = ['label']
        super().__init__(args, split, label_columns, multilabel=False)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"

class SIIMPneumothoraxDataset(BaseXRayDataset):
    def __init__(self, args: Dict[str, Any], split: str):
        label_columns = ['label']
        super().__init__(args, split, label_columns, multilabel=False)

    def _get_glob_pattern(self) -> str:
        return f"{os.path.join(self.image_dir)}/*.jpg"
    
def custom_collate_fn(batch):
    """Custom collate function to handle metadata properly."""
    if len(batch[0]) == 3:  # If metadata exists
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        metadata = [item[2] for item in batch]  # Keep metadata as list of dicts
        return images, labels, metadata
    else:
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return images, labels

def get_dataset(training_args):
    """Get train, validation and test datasets based on configuration."""
    dataset_name = training_args['finetune_dataset']
    label_file = training_args['label_file']
    image_dir = training_args['image_dir']
    seed = training_args.get('seed')
    sampling_ratio = training_args.get('sampling_ratio')
    min_samples_per_label = int(training_args.get('min_samples_per_label')) if training_args.get('min_samples_per_label') else None
    drop_columns = training_args.get('drop_columns')
    no_transform = training_args.get('no_transform', False)
    
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
    
    print("\nArgs:")
    for key, value in args.items():
        print(f"{key}: {value}")
    
    dataset_mapping = {
        "mimic": MIMICCXRDataset,
        "mimic_lt": MIMICCXRLTDataset,
        "chexpert": ChexpertDataset,
        "nih": NIHDataset,
        "nih_lt": NIHLTDataset,
        "covid": COVIDDataset,
        "chexchonet": ChexchonetDataset,
        "tbx11k": TBX11KDataset,
        "rsna": RSNAPneumoniaDataset,
        "siim": SIIMPneumothoraxDataset,
        "vindr-cxr": VindrCXRDataset,
        "vindr-pcxr": VindrPCXRDataset
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError(f"Dataset {dataset_name} not found")
        
    dataset_class = dataset_mapping[dataset_name]
    # Define validation split name based on dataset
    val_split = 'validate' if dataset_name in ['mimic', 'mimic_lt'] else 'valid'
    
    # Create train and test datasets
    train_dataset = dataset_class(args, 'train')
    test_dataset = dataset_class(args, 'test')
    
    # Create validation dataset if applicable
    datasets_without_val = ['chexchonet', 'tbx11k', 'rsna', 'siim', 'vindr-cxr', 'vindr-pcxr']
    val_dataset = dataset_class(args, val_split) if dataset_name not in datasets_without_val else None
    
    return train_dataset, val_dataset, test_dataset