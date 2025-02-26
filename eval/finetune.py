# torchrun --nproc_per_node=3 finetune.py --model_cfg dino_model_config.yaml --training_cfg dino_training_config.yaml

import os
import argparse
import numpy as np
import yaml
import json

import torch
import torch.distributed as dist
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import get_cosine_schedule_with_warmup
from optimi import AdamW
import timm
from timm.models.vision_transformer import VisionTransformer
import wandb

from improved_dataloaders import get_dataset, custom_collate_fn

from metrics import (
    validate_multilabel, validate_singlelabel, init_wandb, log_training_step, log_epoch_metrics, log_test_results
)
from modeling import (
    DINOViTClassifier, DINOViTClassifierSimple, DINOViTClassifierWithLoRA, 
    CLIPTimmClassifier, CLIPTimmClassifierWithLoRA
)
from losses import get_singlelabel_loss, get_multilabel_loss

import warnings
warnings.filterwarnings("ignore")

import logging
from datetime import datetime

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, default='dino_model_config.yaml')
    parser.add_argument('--training_cfg', type=str, default='dino_training_config.yaml')
    return parser.parse_args()

args = arg_parser()

with open(args.model_cfg, 'r') as file:
    if args.model_cfg.endswith('.yaml') or args.model_cfg.endswith('.yml'):
        model_args = yaml.safe_load(file)['student']
    else:
        model_args = json.load(file)

with open(args.training_cfg, 'r') as file:
    training_args = yaml.safe_load(file)

# print("Model_args: ", model_args)
# print("\nTraining args: ", training_args) 
        
# For jupyter notebook
# from dataclasses import dataclass

# @dataclass
# class Args:
#     model_cfg: str = 'dino_model_config.yaml'
#     training_cfg: str = 'dino_training_config.yaml'

# args = Args()

# import os
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "29500"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["RANK"] = "0"  

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def setup_logging(training_args, rank):
    if rank == 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{training_args['run_name']}/{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This will maintain console output
            ]
        )
        # Also capture warnings
        logging.captureWarnings(True)
        # Set warning format to match other log messages
        warning_logger = logging.getLogger('py.warnings')
        warning_logger.handlers = []
        
        return logging.getLogger(__name__)
    return None

def main():
    create_folder(training_args['run_name'])
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    
    logger = setup_logging(training_args, rank)
    
    if rank == 0:
        logger.info(f"Starting training for run name: {training_args['run_name']}")
        logger.info(f"\nModel args: {model_args}")
        logger.info(f"\nTraining args: {training_args}")
    
    train_dataset, val_dataset, test_dataset = get_dataset(training_args)
    metadata_exists = True if len(train_dataset[0]) > 2 else False
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_sampler = DistributedSampler(train_dataset) 
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_args['batch_size'], 
        sampler=train_sampler,
        num_workers=training_args['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    if rank == 0:
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=2*training_args['batch_size'], 
                shuffle=False, 
                num_workers=training_args['num_workers'],
                collate_fn=custom_collate_fn
            )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=2*training_args['batch_size'], 
            shuffle=False, 
            num_workers=training_args['num_workers'],
            collate_fn=custom_collate_fn
        )

    # Model setup
    if training_args['use_timm_vit']:
        if rank == 0:
            logger.info("Using timm vit model")
            logger.info(f"Pretrained Imagenet Backbone: {training_args['imagenet_pretrained_timm_vit']}")
        vit_model = timm.create_model('vit_base_patch16_224.augreg_in21k', 
                                      num_classes=training_args['num_classes'], 
                                      pretrained=training_args['imagenet_pretrained_timm_vit'])
        
        if training_args['freeze'] == 'freeze_all':
            for name, param in vit_model.named_parameters():
                if 'head' not in name:  # Don't freeze classification head
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for param in vit_model.parameters():
                param.requires_grad = True
    elif training_args.get('use_lora', False):
        if rank == 0:
            logger.info("Using LoRA-enabled model")
        if training_args['dino_or_clip'] == "dino":
            vit_model = DINOViTClassifierWithLoRA(model_args, training_args)
        elif training_args['dino_or_clip'] == "clip":
            vit_model = CLIPTimmClassifierWithLoRA(model_args, training_args)
        else:
            raise ValueError(f"Invalid dino_or_clip option: {training_args['dino_or_clip']}")
    else:
        if training_args['dino_or_clip'] == "dino":
            if training_args.get('use_simple_vit', False):
                vit_model = DINOViTClassifierSimple(model_args, training_args)
            else:
                vit_model = DINOViTClassifier(model_args, training_args)
        elif training_args['dino_or_clip'] == "clip":
            vit_model = CLIPTimmClassifier(model_args, training_args)
        else:
            raise ValueError(f"Invalid dino_or_clip option: {training_args['dino_or_clip']}")
    
    vit_model = vit_model.to(rank)
    if training_args['compile']:
        vit_model = torch.compile(vit_model, mode="reduce-overhead")
        print("Compiling the model")

    trainable_params = [param for param in vit_model.parameters() if param.requires_grad]
    if rank == 0:
        logger.info(f"Total params: {sum(p.numel() for p in vit_model.parameters())}")
        logger.info(f"Total trainable params: {sum(p.numel() for p in trainable_params)}")

    optimizer = AdamW(trainable_params, betas=(0.9, 0.95),
                      lr=float(training_args['learning_rate']), 
                      weight_decay=float(training_args['weight_decay']), decouple_lr=True)
    
    vit_model = DDP(vit_model, device_ids=[rank], find_unused_parameters=True)

    warmup_steps = training_args['warmup_epochs'] * len(train_loader)
    total_steps = training_args['num_epochs'] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    total_instance_num = len(train_dataset.data)
    if training_args['multilabel']:
        class_instance_nums = [train_dataset.data[col].sum() for col in train_dataset.label_columns]
        criterion = get_multilabel_loss(loss_type=training_args['loss'], 
                             class_instance_nums=class_instance_nums, 
                             total_instance_num=total_instance_num)
    else:
        class_instance_nums = train_dataset.data[train_dataset.label_column].value_counts()
        criterion = get_singlelabel_loss(loss_type=training_args['loss'],
                                         class_instance_nums=class_instance_nums,
                                         total_instance_num=total_instance_num)

    if rank == 0:
        init_wandb(model_args, training_args)

    best_epoch = 0
    best_val_ap = 0.0
    for epoch in range(training_args['num_epochs']):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        vit_model.train()
        
        if rank == 0:
            logger.info(f"\nEpoch [{epoch+1}/{training_args['num_epochs']}]")
        
        last_5_losses = []
            
        # for batch_idx, (images, labels) in enumerate(train_loader):
        for batch_idx, batch_data in enumerate(train_loader):
            if metadata_exists:
                images, labels, metadata = batch_data
            else:
                images, labels = batch_data
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = vit_model(images)
            if training_args['multilabel']:
                loss = criterion(outputs, labels.float())
            else:
                loss = criterion(outputs, labels)  # For CrossEntropyLoss, labels should be long/int type
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_5_losses.append(loss.item())
            running_loss += loss.item()
                
            if rank == 0 and batch_idx % 5 == 0:
                logger.info(f"Step [{batch_idx}/{len(train_loader)}], LR: {scheduler.get_last_lr()[0]:.8f}, Loss: {np.mean(last_5_losses):.4f}")
                log_training_step(loss, scheduler.get_last_lr()[0], epoch, batch_idx, len(train_loader))
                last_5_losses = []
            
        if rank == 0:
            avg_loss = running_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{training_args['num_epochs']}], Loss: {avg_loss:.4f}")
            
            if val_dataset is not None:
                logger.info("Validating...")
                if training_args['multilabel']:
                    res_val = validate_multilabel(vit_model.module, val_loader, criterion, rank, 
                                                  metadata_exists=metadata_exists, per_class=True)
                else:
                    res_val = validate_singlelabel(vit_model.module, val_loader, criterion, rank, per_class=True)
            else:
                res_val = None
                
            logger.info("Testing...")
            if training_args['multilabel']:
                res_test = validate_multilabel(vit_model.module, test_loader, criterion, rank, 
                                               metadata_exists=metadata_exists, per_class=True)
            else:
                res_test = validate_singlelabel(vit_model.module, test_loader, criterion, rank, per_class=True)
            
            # Save results to CSV
            test_results_df = res_test['results_df']
            
            if res_val is not None:
                best_epoch = epoch if res_val['average_precision'] >= best_val_ap else best_epoch
            else:
                best_epoch = epoch
            
            log_epoch_metrics(avg_loss, res_val, res_test, epoch, multilabel=training_args['multilabel'])
            log_test_results(training_args, test_results_df, epoch)  # Log the results dataframe to wandb

        if rank == 0:
            torch.cuda.empty_cache()

    # if rank == 0:
        # wandb.finish()
        # Keep only the best model
        # for epoch in range(training_args['num_epochs']):
            # if epoch != best_epoch:
                # model_name = f"{training_args['run_name']}/vit_mimic_linear_probe_multilabel_epoch_{epoch+1}.pth"
                # os.remove(model_name)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()