import datetime
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, accuracy_score
)
import wandb
from tqdm import tqdm
import logging
import pandas as pd

def validate_singlelabel(model, val_loader, criterion, device, per_class=False):
    logger = logging.getLogger(__name__)
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for inputs, labels, metadata in tqdm(val_loader, disable=device != 0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            all_metadata.extend(metadata)

    avg_val_loss = val_loss / len(val_loader)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    # Get predicted class indices
    _, preds = torch.max(all_outputs, 1)
    
    # Convert to numpy for sklearn metrics
    preds = preds.numpy()
    labels = all_labels.numpy()
    probs = torch.softmax(all_outputs, dim=1).numpy()

    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro') 
    recall = recall_score(labels, preds, average='macro')
    roc_auc = roc_auc_score(labels, probs, average='macro', multi_class='ovr')

    logger.info(f"Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, "
                f"F1-Score: {f1:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Create results dataframe with metadata
    results_df = pd.DataFrame({
        'path': [m['path'] for m in all_metadata],
        'true_label': labels,
        'predicted_label': preds
    })

    # Add probabilities for each class
    for i in range(probs.shape[1]):
        results_df[f'prob_class_{i}'] = probs[:, i]

    return {
        'loss': avg_val_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'results_df': results_df
    }
    

def validate_multilabel(model, val_loader, criterion, device, metadata_exists=False, per_class=False):
    logger = logging.getLogger(__name__)
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        if metadata_exists:
            for inputs, labels, metadata in tqdm(val_loader, disable=device != 0):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                all_metadata.extend(metadata)  # Collect metadata
        else:
            for inputs, labels in tqdm(val_loader, disable=device != 0):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

    avg_val_loss = val_loss / len(val_loader)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    all_outputs = torch.sigmoid(all_outputs)
    preds = all_outputs > 0.5 

    f1 = f1_score(all_labels.numpy(), preds.numpy(), average='macro')
    precision = precision_score(all_labels.numpy(), preds.numpy(), average='macro')
    recall = recall_score(all_labels.numpy(), preds.numpy(), average='macro')
    accuracy = accuracy_score(all_labels.numpy(), preds.numpy())
    roc_auc = roc_auc_score(all_labels.numpy(), all_outputs.numpy(), average='macro')
    ap = average_precision_score(all_labels.numpy(), all_outputs.numpy(), average='macro')
    

    logger.info(f"Loss: {avg_val_loss:.4f}, Mean Average Precision: {ap:.4f}, "
                f"F1-Score: {f1:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
    
    # Create results dataframe with metadata
    # Initialize with path which should always be present
    if metadata_exists:
        results_df = pd.DataFrame({
            'path': [m['path'] for m in all_metadata]
        })
    
        # Add age and sex if they exist in the dataset
        if 'age' in val_loader.dataset.data.columns:
            results_df['age'] = [m['age'] for m in all_metadata]
        if 'sex' in val_loader.dataset.data.columns:
            results_df['sex'] = [m['sex'] for m in all_metadata]
        if 'race' in val_loader.dataset.data.columns:
            results_df['race'] = [m['race'] for m in all_metadata]
    
    
        for i, col in enumerate(val_loader.dataset.label_columns):
            results_df[f'{col}_pred'] = all_outputs.numpy()[:, i]
            results_df[f'{col}_true'] = all_labels[:, i]
    else:
        for i, col in enumerate(val_loader.dataset.label_columns):
            results_df = pd.DataFrame()
            results_df[f'{col}_pred'] = all_outputs.numpy()[:, i]
            results_df[f'{col}_true'] = all_labels[:, i]

    out_dict = {
        'loss': avg_val_loss,
        'average_precision': ap,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'class_metrics': {},
        'results_df': results_df  # Add results dataframe to output
    }
    
    if per_class:
        for i in range(all_labels.shape[1]):
            f1 = f1_score(all_labels[:, i].numpy(), preds[:, i].numpy())
            precision = precision_score(all_labels[:, i].numpy(), preds[:, i].numpy())
            recall = recall_score(all_labels[:, i].numpy(), preds[:, i].numpy())
            accuracy = accuracy_score(all_labels[:, i].numpy(), preds[:, i].numpy())
            roc = roc_auc_score(all_labels[:, i].numpy(), all_outputs[:, i].numpy())
            ap = average_precision_score(all_labels[:, i].numpy(), all_outputs[:, i].numpy())
            label_sum = all_labels[:, i].sum().item()
            
            out_dict['class_metrics'][i] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'roc_auc': roc,
                'average_precision': ap,
                'label_sum': label_sum
            }
            logger.info(f"Class {i}: Label Sum: {label_sum}, ROC-AUC: {roc:.4f}, Average Precision (AP): {ap:.4f}, \
                        F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
    del all_outputs, all_labels, preds
    torch.cuda.empty_cache()

    return out_dict

def init_wandb(model_args, training_args):
    logger = logging.getLogger(__name__)
    if training_args['run_name'] == "":
        run_name = f"{training_args['wandb_project']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_name = training_args['run_name']
    logger.info(f"Initializing W&B run: {run_name}")
    config = {"model": model_args,"training": training_args}
    wandb.init(
        project=training_args['wandb_project'],
        name=run_name,
        config=config
    )

def log_training_step(loss, lr, epoch, batch_idx, train_loader_len):
    wandb.log({
        "train/loss": loss.item(),
        "train/learning_rate": lr,
    }, step=epoch * train_loader_len + batch_idx)

def log_epoch_metrics(train_loss, val_results, test_results, epoch, multilabel=False):
    # Log training metrics
    wandb.log({
        "train/epoch_loss": train_loss,
        "epoch": epoch
    })
    
    if multilabel:
        if val_results is not None:
            wandb.log({
                "val/loss": val_results['loss'],
                "val/average_precision": val_results['average_precision'],
                "val/f1": val_results['f1'],
                "val/precision": val_results['precision'],
                "val/recall": val_results['recall'],
                "val/roc_auc": val_results['roc_auc'],
                "epoch": epoch
            })
    
        wandb.log({
            "test/loss": test_results['loss'],
            "test/average_precision": test_results['average_precision'],
            "test/f1": test_results['f1'],
            "test/precision": test_results['precision'],
            "test/recall": test_results['recall'],
            "test/roc_auc": test_results['roc_auc'],
            "epoch": epoch
        })
    else:
        # Add single-label metrics logging
        if val_results is not None:
            wandb.log({
                "val/loss": val_results['loss'],
                "val/accuracy": val_results['accuracy'],
                "val/f1_score": val_results['f1_score'],
                "val/precision": val_results['precision'],
                "val/recall": val_results['recall'],
                "val/roc_auc": val_results['roc_auc'],
                "epoch": epoch
            })

        wandb.log({
            "test/loss": test_results['loss'],
            "test/accuracy": test_results['accuracy'], 
            "test/f1_score": test_results['f1_score'],
            "test/precision": test_results['precision'],
            "test/recall": test_results['recall'],
            "test/roc_auc": test_results['roc_auc'],
            "epoch": epoch
        })
        
    if val_results is not None:
        # Log per-class metrics
        if 'class_metrics' in val_results.keys():
            for class_idx, metrics in val_results['class_metrics'].items():
                wandb.log({
                    f"val_class_{class_idx}/f1": metrics['f1'],
                    f"val_class_{class_idx}/precision": metrics['precision'],
                    f"val_class_{class_idx}/recall": metrics['recall'],
                    f"val_class_{class_idx}/accuracy": metrics['accuracy'],
                    f"val_class_{class_idx}/roc_auc": metrics['roc_auc'],
                    f"val_class_{class_idx}/average_precision": metrics['average_precision'],
                    "epoch": epoch
                })
            
            for class_idx, metrics in test_results['class_metrics'].items():
                wandb.log({
                    f"test_class_{class_idx}/f1": metrics['f1'],
                    f"test_class_{class_idx}/precision": metrics['precision'],
                    f"test_class_{class_idx}/recall": metrics['recall'],
                    f"test_class_{class_idx}/accuracy": metrics['accuracy'],
                    f"test_class_{class_idx}/roc_auc": metrics['roc_auc'],
                    f"test_class_{class_idx}/average_precision": metrics['average_precision'],
                    "epoch": epoch
                })
        
def log_test_results(training_args, test_results, epoch):
    if epoch == training_args['num_epochs'] - 1:
        test_results.to_csv(f"{training_args['run_name']}/test_predictions.csv", index=False)
        if 'age' in test_results.columns:
            test_results['age'] = test_results['age'].astype(str)
        
        wandb.log({
            "test_results": wandb.Table(dataframe=test_results),
            "epoch": epoch
        })
