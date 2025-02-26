Things to do before running evals:

```bash
git clone https://github.com/raunak-agarwal/dinov2 && cd dinov2 
mv requirements-eval.txt requirements.txt
pip install -U -r requirements.txt scikit-learn wandb numpy==1.26.4 nvitop timm transformers peft torch-optimi -e .[dev] open_clip_torch[training]


wget -O mimic.zip mimic-url
mkdir mimic && unzip mimic.zip -d mimic/
mkdir chexpert && unzip chexpert.zip -d chexpert/

wget -O training_99999.pth dino-checkpoint-url
wget -O cxr-bert-epoch_50.pt cxr-bert-checkpoint-url


mkdir -p checkpoints data/ && mv mimic-cxr-2.0.0-merged.csv data/ && tar -I zstd -xvf mimic-cxr-images.tar.zst -C data && unzip training_231999.zip -d checkpoints/ && rm -f wget-log*


git clone this_repo

export XFORMERS_DISABLED=1 # if xformers is not available (usually on older versions of cuda)

add --master_port=free_port if training multiple models 
```
