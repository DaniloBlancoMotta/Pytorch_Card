import torch
from src.pipelines.training_pipeline import run_training
import os

def main():
    # Final production paths
    train_dir = 'data/01-raw/train'
    val_dir = 'data/01-raw/val'
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Directory {train_dir} not found. Please setup data first.")
        return

    model, classes = run_training(train_dir, val_dir, num_epochs=10)

    # Save artifact
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes
    }, 'model_checkpoint.pth')
    print("Model saved to model_checkpoint.pth")

if __name__ == "__main__":
    main()
