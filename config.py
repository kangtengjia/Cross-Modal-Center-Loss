import argparse
import os
import torch

def get_train_config():
    # Training settings
    parser = argparse.ArgumentParser(description='Cross Modal Retrieval for Point Cloud, Mesh, and Image Models')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='ModelNet40', 
                       help='ModelNet10 or ModelNet40')
    parser.add_argument('--dataset_dir', type=str, default='./dataset/', 
                       help='Path to dataset directory')
    parser.add_argument('--num_classes', type=int, default=40, 
                       help='Number of classes (10 or 40)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=96, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, 
                       help='Number of epochs to train')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=20000, 
                       help='Iterations to decrease learning rate')
    parser.add_argument('--lr_center', type=float, default=0.001, 
                       help='Learning rate for center loss')
    parser.add_argument('--momentum', type=float, default=0.9, 
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-3, 
                       help='Weight decay')
    
    # Model parameters
    parser.add_argument('--num_points', type=int, default=1024, 
                       help='Number of points in point cloud')
    parser.add_argument('--weight_center', type=float, default=10, 
                       help='Weight for center loss')
    
    # Logging and saving parameters
    parser.add_argument('--per_save', type=int, default=5000, 
                       help='Iterations to save model')
    parser.add_argument('--per_print', type=int, default=100, 
                       help='Iterations to print loss and accuracy')
    parser.add_argument('--save', type=str, default='./checkpoints/ModelNet40', 
                       help='Path to save model checkpoints')
    
    # GPU parameters
    parser.add_argument('--gpu_id', type=str, default='1', 
                       help='GPU IDs to use')
    parser.add_argument('--log', type=str, default='log/', 
                       help='Path to log information')

    parser.add_argument('--k', type=int, default=20, 
                       help='The number of nearest neighbors in DGCNN')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_train_config()
    print("Training configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"GPU IDs: {args.gpu_id}")