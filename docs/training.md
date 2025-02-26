### Training DINOv2


#### Environment setup. Tested on runpod and vast.ai on 4-8 A100/H100 GPUs.
```bash
cd workspace
sudo apt update && sudo apt upgrade -y && sudo apt install tmux htop zstd curl nano -y

git clone https://github.com/raunak-agarwal/dinov2
cd dinov2
pip install -r requirements.txt
pip install -e .[dev] 

cd workspace
wget -O dino_train_images_flat.tar.zst dataset_url

export WANDB_API_KEY="" && 

export XFORMERS_DISABLED=1

mkdir datasets/
tar -I zstd -xvf dino_train_images_flat.tar.zst -C /workspace/datasets/

torchrun --nproc_per_node=4 dinov2/train/train.py --config-file=dinov2/configs/train/vitl14-a100x4.yaml --output-dir=/workspace/dinov2/checkpoints/dinov2-vitl14-all

torchrun --nproc_per_node=4 dinov2/train/train.py --config-file=dinov2/configs/train/vitb14-a100.yaml --output-dir=/workspace/dinov2/checkpoints/dinov2-vitb14-all

torchrun --nproc_per_node=4 dinov2/train/train.py --config-file=dinov2/configs/train/vitb16-h100x4.yaml --output-dir=/home/ubuntu/workspace/dinov2/checkpoints/dinov2-vitb16-all



```

### Training CLIP

Tested on a single 4090.

```bash
apt update && apt upgrade -y && apt install zstd htop unzip tmux -y

wget -O train-data.zip train-data-url

wget -O test-sample.tar test-sample-url

wget -O mimic-test.tar mimic-test-url

mkdir train && unzip train-data.zip -d train/

git clone https://github.com/mlfoundations/open_clip.git

cp custom-config.json open_clip/src/open_clip/model_configs/

cd open_clip && make install && pip install -r requirements-training.txt && pip install nvitop wandb flash-attn git+https://github.com/huggingface/transformers.git

pip install 'open_clip_torch[training]'

export WANDB_API_KEY=wandb_api_key

cd open_clip/src
torchrun --nproc_per_node 1 -m open_clip_train.main \
    --train-data '/workspace/train/{0000..2172}.tar' \
    --val-data '/workspace/mimic-test.tar' \
    --train-num-samples 838785 \
    --val-num-samples 5159 \
    --dataset-type webdataset \
    --model "custom-config" \
    --batch-size 200 \
    --precision amp \
    --warmup 12000 \
    --lr 5e-4 \
    --epochs 50 \
    --workers 24 \
    --grad-clip-norm 1.0 \
    --image-mean 0.4958348274 0.4958348274 0.4958348274 \
    --image-std 0.2771022319 0.2771022319 0.2771022319 \
    --force-image-size 224 \
    --report-to wandb \
    --wandb-project-name "clip-training"

```