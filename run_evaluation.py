#!/usr/bin/env python3
"""
Script to run evaluation of the trained Cross-Modal-Center-Loss model.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Evaluate Cross-Modal-Center-Loss model')
    parser.add_argument('--dataset', type=str, default='ModelNet40', 
                       help='ModelNet10 or ModelNet40')
    parser.add_argument('--model_folder', type=str, default='ModelNet40', 
                       help='Model folder name')
    parser.add_argument('--iterations', type=int, default=55000, 
                       help='Number of iterations the model was trained for')
    parser.add_argument('--gpu_id', type=str, default='0', 
                       help='GPU IDs to use for evaluation')
    parser.add_argument('--save', type=str, default='extracted_features/ModelNet40', 
                       help='Path to save extracted features')
    
    args = parser.parse_args()
    
    print(f"Running evaluation for {args.dataset}")
    print(f"Model folder: {args.model_folder}")
    print(f"Trained for {args.iterations} iterations")
    print(f"Using GPUs: {args.gpu_id}")
    print(f"Saving features to: {args.save}")
    
    # Run the evaluation script
    cmd = f"python evaluate_retrieval.py --dataset {args.dataset} --model_folder {args.model_folder} --iterations {args.iterations} --gpu_id {args.gpu_id} --save {args.save}"
    print(f"Executing: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()