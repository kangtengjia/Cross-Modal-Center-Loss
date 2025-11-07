# Cross-Modal Center Loss

This repository implements a cross-modal retrieval approach for 3D shapes using images, point clouds, and meshes. The implementation is based on the paper "Cross-Modal Center Loss for 3D Cross-Modal Retrieval" by Yawei Li et al.

## Project Structure

```
├── config.py              # Training configuration
├── train.py               # Main training script
├── evaluate_retrieval.py  # Evaluation script
├── run_evaluation.py      # Helper script to run evaluations
├── models/                # Neural network architectures
│   ├── corrnet.py         # CorrNet model combining all modalities
│   ├── dgcnn.py           # Point cloud network
│   ├── meshnet.py         # Mesh network
│   └── resnet.py          # Image network
├── tools/
│   ├── dataloader.py      # Data loading utilities
│   └── test_dataloader.py # Test data loading utilities
├── dataset/               # Dataset directory (to be populated)
└── checkpoints/           # Model checkpoints (created during training)
```

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (for training)
- Required packages in requirements.txt

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

Download and organize datasets in the `dataset/` directory:
- ModelNet40 point clouds and images
- ModelNet40 meshes

## Training

To train the model, run:
```
python train.py --dataset ModelNet40 --num_classes 40 --batch_size 96 --epochs 1000
```

Key training parameters can be adjusted in `config.py`.

## Evaluation

To evaluate a trained model:
```
python run_evaluation.py --dataset ModelNet40 --model_folder ModelNet40 --iterations 55000
```

## Configuration

Training parameters can be modified in `config.py`:
- Dataset settings (ModelNet40/ModelNet10)
- Training hyperparameters (learning rates, batch size, etc.)
- Model architecture parameters
- GPU configuration

## Results

The model achieves state-of-the-art performance on cross-modal retrieval tasks between 3D shapes represented as point clouds, meshes, and images.

## Citation

If you find this work useful, please cite the original paper:
```
@article{li2021cross,
  title={Cross-Modal Center Loss for 3D Cross-Modal Retrieval},
  author={Li, Yawei and Zhang, Ronghang and Zhai, Guangtao and Min, Xiongkuo and Zhao, Yao and Yang, Xiaokang},
  journal={IEEE Transactions on Image Processing},
  year={2021}
}
```