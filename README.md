# Comparing DINOv2 and CLIP for Chest X-Rays


The code is structured as follows:

1. Data Processing
   - Code for processing the data is in [`data_processing`](data_processing)
   - Detailed instructions are available in [`docs/preprocessing.md`](docs/preprocessing.md)
2. Pretraining
   - For DINOv2 training, we create a fork which adds a custom dataloader. It is available [here](https://github.com/raunak-agarwal/dinov2)
     - The config files and logs for training this model are available in [`pretraining/dino`](pretraining/dino)
   - For training CLIP models, we use the official OpenCLIP implementation. Available [here](https://github.com/mlfoundations/open_clip)>
     - The datasets are packaged in [webdataset](https://github.com/webdataset/webdataset) format.
     - The config files and logs for training this model are available in [`pretraining`](pretraining)
   - More details on training DINOv2 and CLIP models are available in [`docs/training.md`](docs/training.md)
3. Evaluation
   - To pre-compute embeddings for KNN and Linear Probing, use [`eval/create_embeddings.py`](eval/create_embeddings.py)
   - KNN code is in [`eval/knn_single_label.py`](eval/knn_single_label.py) and [`eval/knn_multi_label.py`](eval/knn_multi_label.py)
   - Linear Probing code is in [`eval/fast_linearprobe.py`](eval/fast_linearprobe.py)
   - Image-Retrieval code is in [`eval/image-knn.py`](eval/image-knn.py)
   - Image-Text Retrieval code is in [`eval/clip/image-text-retrieval-evals.ipynb`](eval/clip/image-text-retrieval-evals.ipynb)
   - Code for full-finetuning is in [`eval/finetune.py`](eval/finetune.py)
   - Model definintions are available in [`eval/modeling.py`](eval/modeling.py)
   - Metric computation code is in [`eval/metrics.py`](eval/metrics.py)
   - More details on evaluating the models are available in [`docs/evaluation.md`](docs/evaluation.md)
   - Code for creating loss curves is in [`pretraining/figures/`](pretraining/figures/)