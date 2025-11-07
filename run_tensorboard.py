#!/usr/bin/env python3
"""
Script to start TensorBoard for visualizing training progress.
"""

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Start TensorBoard for Cross-Modal Center Loss training')
    parser.add_argument('--logdir', type=str, default='checkpoints/ModelNet40/summary', 
                       help='Directory with TensorBoard logs')
    parser.add_argument('--port', type=int, default=6006, 
                       help='Port to run TensorBoard on')
    
    args = parser.parse_args()
    
    print(f"Starting TensorBoard with logs from: {args.logdir}")
    print(f"Access TensorBoard at: http://localhost:{args.port}")
    
    # Check if the log directory exists
    if not os.path.exists(args.logdir):
        print(f"Warning: Log directory {args.logdir} does not exist yet.")
        print("TensorBoard will start but may not show any data until training begins.")
    
    # Start TensorBoard
    try:
        cmd = f"tensorboard --logdir {args.logdir} --port {args.port}"
        print(f"Executing: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting TensorBoard: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")

if __name__ == "__main__":
    main()