from datetime import datetime
import logging
import torch
import torch.nn as nn
import os
from collections import OrderedDict
import json
import timm
from timm.layers import Mlp, to_2tuple
from peft import LoraConfig, get_peft_model

# from timm.models.vision_transformer import VisionTransformer

import dinov2.dinov2.models.vision_transformer as vits

from ml_decoder import MLDecoder
from attention_pool import AttentionPoolLatent

# from timm.layers.attention_pool2d import AttentionPool2d
# from dinov2.dinov2.eval.utils import ModelWithNormalize\
# from timm.layers.classifier import NormMlpClassifierHead, ClassifierHead
# from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


def load_dino_teacher_backbone_weights(model_path):
    print(f"Loading teacher backbone weights from {model_path}")
    model = torch.load(model_path, map_location='cpu')['teacher']
    teacher_backbone_keys = [k for k in model.keys() if k.startswith('backbone.')]
    teacher_backbone_weights = {k.replace("teacher.","").replace("backbone.","") : model[k] for k in teacher_backbone_keys}
    teacher_backbone_weights = {key: value for key, value in teacher_backbone_weights.items() if not key.startswith(("dino", "ibot"))}
    del model
    torch.cuda.empty_cache()
    return teacher_backbone_weights

def load_clip_weights(model_path):
    print(f"Loading clip weights from {model_path}")
    weights = torch.load(model_path, map_location='cpu')
    state_dict = weights['state_dict']
    state_dict = {k.replace("visual.", "") :v for k, v in state_dict.items() if "visual" in k}
    torch.cuda.empty_cache()
    return state_dict



class DINOEmbeddingModel(nn.Module):
    def __init__(self, model_args, training_args):
        super(DINOEmbeddingModel, self).__init__()
        self.vit_kwargs = dict(
            img_size=224,
            patch_size=model_args['patch_size'],
            init_values=model_args['layerscale'],
            ffn_layer=model_args['ffn_layer'],
            block_chunks=model_args['block_chunks'],
            qkv_bias=model_args['qkv_bias'],
            proj_bias=model_args['proj_bias'],
            ffn_bias=model_args['ffn_bias'],
            num_register_tokens=model_args['num_register_tokens'],
            interpolate_offset=model_args['interpolate_offset'],
            interpolate_antialias=model_args['interpolate_antialias'],
            drop_path_rate=model_args['drop_path_rate'],
            drop_path_uniform=model_args['drop_path_uniform']
        )
        self.model = vits.__dict__[model_args['arch']](**self.vit_kwargs)
        self.model.load_state_dict(load_dino_teacher_backbone_weights(training_args['model_path']))
        self.model.eval()
        # self.normalize = training_args['normalize']

    def forward(self, x, get_intermediate_layers=False):
        if get_intermediate_layers:
            x = self.model.get_intermediate_layers(x, 1, return_class_token=True)
            patch_mean = x[0][0].mean(dim=1)  # Average over patches: (batch, 768)
            cls_token = x[0][1]  # Class token: (batch, 768) 
            x = torch.cat([cls_token, patch_mean], dim=1) # Final embedding: (batch, 1536)
        else:
            x = self.model(x) # (batch, 768)
        # x = x.mean(dim=0)
        # if self.normalize:
        #     return nn.functional.normalize(x, dim=0, p=2)
        # x = self.model(x)
        return x
    



class DINOViTClassifierSimple(nn.Module):
    def __init__(self, model_args, training_args):
        super(DINOViTClassifierSimple, self).__init__()
        self.vit_kwargs = dict(
            img_size=224,
            patch_size=model_args['patch_size'],
            init_values=model_args['layerscale'],
            ffn_layer=model_args['ffn_layer'],
            block_chunks=model_args['block_chunks'],
            qkv_bias=model_args['qkv_bias'],
            proj_bias=model_args['proj_bias'],
            ffn_bias=model_args['ffn_bias'],
            num_register_tokens=model_args['num_register_tokens'],
            interpolate_offset=model_args['interpolate_offset'],
            interpolate_antialias=model_args['interpolate_antialias'],
            drop_path_rate=model_args['drop_path_rate'],
            drop_path_uniform=model_args['drop_path_uniform']
        )
        
        self.vit = vits.__dict__[model_args['arch']](**self.vit_kwargs)
        if training_args['model_path']:
            self.vit.load_state_dict(load_dino_teacher_backbone_weights(training_args['model_path']))
        
        if hasattr(self.vit, 'mask_token'):
            delattr(self.vit, 'mask_token')
        
        self.classifier = nn.Linear(768, training_args['num_classes'])

        if training_args['freeze'] == 'freeze_all':
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.parameters():
                param.requires_grad = True

    def forward(self, x, is_training=False):        
        x = self.vit(x, is_training=is_training)
        if is_training:
            patch_mean = x['x_norm_patchtokens'].mean(dim=1) # (n_tokens, 768) becomes (1, 768)
            x = torch.cat([x['x_norm_clstoken'], patch_mean], dim=1) # (1, 768 + 768)
        x = self.classifier(x)
        return x

class DINOViTClassifier(nn.Module):
    def __init__(self, model_args, training_args):
        super(DINOViTClassifier, self).__init__()
        self.vit_kwargs = dict(
            img_size=224,
            patch_size=model_args['patch_size'],
            init_values=model_args['layerscale'],
            ffn_layer=model_args['ffn_layer'],
            block_chunks=model_args['block_chunks'],
            qkv_bias=model_args['qkv_bias'],
            proj_bias=model_args['proj_bias'],
            ffn_bias=model_args['ffn_bias'],
            num_register_tokens=model_args['num_register_tokens'],
            interpolate_offset=model_args['interpolate_offset'],
            interpolate_antialias=model_args['interpolate_antialias'],
            drop_path_rate=model_args['drop_path_rate'],
            drop_path_uniform=model_args['drop_path_uniform']
        )
        
        self.vit = vits.__dict__[model_args['arch']](**self.vit_kwargs)
        if training_args['model_path']:
            self.vit.load_state_dict(load_dino_teacher_backbone_weights(training_args['model_path']))
        self.layers = training_args['layers']
        self.pool_avg = training_args['pool_avg']
        self.latent_len = training_args['latent_len']
        if training_args['pool'] == "AttentionPoolLatent":
            if self.pool_avg:
                self.pool = AttentionPoolLatent(in_features=768, latent_len=training_args['latent_len'], feat_size=260, pos_embed="abs", pool_type="avg")
            else:
                self.pool = AttentionPoolLatent(in_features=768, latent_len=training_args['latent_len'], feat_size=260, pos_embed="abs", pool_type=None)
        else:
            self.pool = None
        
        if hasattr(self.vit, 'mask_token'):
            delattr(self.vit, 'mask_token')
        
        # classifier_input_dim = 768 if training_args['pool'] == "AttentionPoolLatent" else (1 + self.layers) * 768
        classifier_input_dim = 768 if training_args['pool'] == "AttentionPoolLatent" else (1 + 1) * 768
        if training_args['classification_head'] == "linear":
            print(f"Linear classifier input dim: {classifier_input_dim}")
            self.classifier = nn.Linear(classifier_input_dim, training_args['num_classes'])
        elif training_args['classification_head'] == "MLDecoder":
            print(f"MLDecoder classifier input dim: {classifier_input_dim}")
            self.classifier = MLDecoder(num_classes=training_args['num_classes'], 
                                        initial_num_features=classifier_input_dim, 
                                        num_of_groups=-1, decoder_embedding=768, zsl=0)

        if training_args['freeze'] == 'freeze_all':
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.parameters():
                param.requires_grad = True

    def forward(self, x):        
        x = self.vit.get_intermediate_layers(x, self.layers, return_class_token=True)
        if not self.pool:
            # See here: https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/classifiers.py
            # linear_input = torch.cat([
            #             x[0][1],
            #             x[1][1],
            #             x[2][1],
            #             x[3][1],
            #             x[3][0].mean(dim=1),
            #     ], dim=1) # shape: 1x3840
            linear_input = torch.cat([
                        x[3][1],
                        x[3][0].mean(dim=1),
                ], dim=1) # shape: 1x1536
        else:
            class_tokens = [x[i][1].unsqueeze(1) for i in range(4)]  # List of 4 tensors, each (batch_size, 1, 768)
            patch_tokens = x[3][0]  # (batch_size, num_patches (256), 768)
            combined = torch.cat([*class_tokens, patch_tokens], dim=1)  # Output shape: (batch_size, 4+256, 768)
            linear_input = self.pool(combined) # Output shape: (batch_size, 768) if pool_avg is True, else (batch_size, latent_len, 768)
        
        if isinstance(self.classifier, MLDecoder):
            if self.pool_avg:
                linear_input = linear_input.unsqueeze(1) # Output shape: (batch_size, 1, 768)

        x = self.classifier(linear_input) # Output shape: (batch_size, 14)
        return x


# class DINOViTClassifierWithLoRA(DINOViTClassifier):
#     def __init__(self, model_args, training_args):
#         super(DINOViTClassifierWithLoRA, self).__init__(model_args, training_args)
        
#         # Apply LoRA configuration
#         self.lora_config = LoraConfig(
#             r=48,
#             lora_alpha=16,
#             target_modules=["qkv", "fc1", "fc2"],
#             lora_dropout=0.1,
#             bias="lora_only",
#             modules_to_save=["classifier"],
#         )
#         self.vit = get_peft_model(self.vit, self.lora_config)

#     def save_model(self, save_path):
#         save_dict = {
#             "lora_state_dict": self.vit.state_dict(),
#             "classifier_state_dict": self.classifier.state_dict(),
#             "lora_config": self.lora_config,
#             "model_args": self.vit_kwargs,
#             "training_args": {
#                 'layers': self.layers,
#                 'pool': 'AttentionPoolLatent' if self.pool else None,
#                 'classification_head': 'MLDecoder' if isinstance(self.classifier, MLDecoder) else 'linear',
#                 'num_classes': self.classifier.out_features if isinstance(self.classifier, nn.Linear) 
#                               else self.classifier.num_classes
#             }
#         }
#         torch.save(save_dict, save_path)

#     @classmethod
#     def load_model(cls, checkpoint_path, device='cuda'):
#         checkpoint = torch.load(checkpoint_path, map_location=device)
        
#         model = cls(checkpoint["model_args"], checkpoint["training_args"])
#         model.vit.load_state_dict(checkpoint["lora_state_dict"])
#         model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
#         return model.to(device)



class DINOViTClassifierWithLoRA(DINOViTClassifier):
    def __init__(self, model_args, training_args):
        super(DINOViTClassifierWithLoRA, self).__init__(model_args, training_args)
        
        # Apply LoRA configuration
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["head"], # Changed from classifier to head to match DINO model
        )
        
        # Create base DINO model
        self.vit = vits.__dict__[model_args['arch']](**self.vit_kwargs)
        
        # Load DINO weights if model path provided
        if training_args['model_path']:
            self.vit.load_state_dict(load_dino_teacher_backbone_weights(training_args['model_path']))
        
        # Add linear classifier head
        self.vit.head = nn.Linear(self.vit.embed_dim, training_args['num_classes'])
        
        # Apply LoRA
        self.vit = get_peft_model(self.vit, self.lora_config)

    def save_model(self, save_path):
        """Save the model checkpoint"""
        # Only save essential LoRA config parameters
        config_dict = {
            'r': self.lora_config.r,
            'lora_alpha': self.lora_config.lora_alpha,
            'target_modules': self.lora_config.target_modules,
            'lora_dropout': self.lora_config.lora_dropout,
            'bias': self.lora_config.bias,
            'modules_to_save': self.lora_config.modules_to_save
        }
        
        save_dict = {
            "lora_state_dict": self.vit.state_dict(),
            "lora_config_dict": config_dict,
            "model_args": self.vit_kwargs,
            "training_args": {
                'layers': self.layers,
                'pool': 'AttentionPoolLatent' if self.pool else None,
                'num_classes': self.vit.head.out_features
            }
        }
        torch.save(save_dict, save_path)

    @classmethod
    def load_model(cls, checkpoint_path, device='cuda'):
        """Load the model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create new model instance
        model = cls(checkpoint["model_args"], checkpoint["training_args"])
        
        # Recreate LoRA config and apply
        lora_config = LoraConfig(**checkpoint["lora_config_dict"])
        model.vit = get_peft_model(model.vit, lora_config)
        
        # Load saved weights
        model.vit.load_state_dict(checkpoint["lora_state_dict"])
        
        return model.to(device)


class CLIPTimmModelBackbone(nn.Module):
    """ timm model adapter
    Model code exactly as this https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py
    Config from here https://github.com/mlfoundations/open_clip/blob/b2f1403605aade5a004434076246b6bc741aa47d/src/open_clip/model.py#L27
    """
    def __init__(
            self,
            model_name,
            embed_dim,
            image_size=224,
            pool='avg',
            proj='linear',
            proj_bias=False,
            drop=0.,
            drop_path=None,
            patch_drop=None,
            pretrained=False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please `pip install timm` to use timm models.")
        self.image_size = to_2tuple(image_size)

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == 'none' else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # # Add custom pooling to head
        # if pool == 'abs_attn':
        #     head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
        #     prev_chs = embed_dim
        # elif pool == 'rot_attn':
        #     head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x
    

class CLIPTimmClassifier(nn.Module):
    def __init__(self, model_args, training_args):
        super(CLIPTimmClassifier, self).__init__()
        self.backbone = CLIPTimmModelBackbone(
            model_name=model_args['model_cfg']['vision_cfg']['timm_model_name'],
            embed_dim=model_args['model_cfg']['embed_dim'], 
            proj=model_args['model_cfg']['vision_cfg']['timm_proj'],
            drop_path=model_args['model_cfg']['vision_cfg']['timm_drop_path']
        )
        
        self.backbone.load_state_dict(load_clip_weights(training_args['model_path']))
        torch.cuda.empty_cache()

        classifier_input_dim = model_args['model_cfg']['embed_dim']
        if training_args['classification_head'] == "linear":
            print(f"Linear classifier input dim: {classifier_input_dim}")
            self.classifier = nn.Linear(classifier_input_dim, training_args['num_classes'])
        elif training_args['classification_head'] == "MLDecoder":
            print(f"MLDecoder classifier input dim: {classifier_input_dim}")
            self.classifier = MLDecoder(num_classes=training_args['num_classes'], 
                                      initial_num_features=classifier_input_dim, 
                                      num_of_groups=-1, decoder_embedding=768, zsl=0)

        if training_args['freeze'] == 'freeze_all':
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(self.classifier, MLDecoder):
            x = x.unsqueeze(1)
        x = self.classifier(x)
        return x
    
    

class CLIPTimmClassifierWithLoRA(CLIPTimmClassifier):
    def __init__(self, model_args, training_args):
        super(CLIPTimmClassifierWithLoRA, self).__init__(model_args, training_args)
        self.model_args = model_args
        self.training_args = training_args
        # Apply LoRA configuration
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["qkv", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["classifier"],
        )
        self.backbone = get_peft_model(self.backbone, self.lora_config)

    def save_model(self, save_path):
        save_dict = {
            "lora_state_dict": self.backbone.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "lora_config": self.lora_config,
            "model_args": self.model_args,
            "training_args": self.training_args
        }
        torch.save(save_dict, save_path)

    @classmethod
    def load_model(cls, checkpoint_path, device='cuda'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = cls(checkpoint["model_args"], checkpoint["training_args"])
        model.backbone.load_state_dict(checkpoint["lora_state_dict"])
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        return model.to(device)
    
"""
For LORA, do the following:

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=48,
    lora_alpha=16,
    target_modules=["qkv", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["classifier"],
)
feature_model = get_peft_model(feature_model, config)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(feature_model)

"""

