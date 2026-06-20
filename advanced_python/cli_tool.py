import argparse
import sys
from typing import List


"""
LESSON 31: Building Command Line Interfaces (CLI)

Goal:
    Learn how to create professional command-line tools.
    Very useful for AI/ML projects, automation, and deployment.
"""


# ==================== 1. MAIN APPLICATION LOGIC ====================
def train_model(epochs: int, learning_rate: float, model_name: str):
    """Simulate training a model."""
    print(f"🚀 Training {model_name} for {epochs} epochs with LR={learning_rate}")
    print("✅ Model trained successfully!")


def predict(input_data: str, model_path: str):
    """Simulate making a prediction."""
    print(f"🔮 Making prediction on: {input_data}")
    print(f"Using model: {model_path}")
    print("✅ Prediction: Positive Sentiment (0.92 confidence)")


def evaluate(dataset: str):
    """Simulate model evaluation."""
    print(f"📊 Evaluating model on dataset: {dataset}")
    print("Accuracy: 0.89 | F1 Score: 0.87 | ROC-AUC: 0.91")
    print("✅ Evaluation completed!")


# ==================== 2. CLI SETUP ====================
def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="AI Training & Prediction CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ==================== TRAIN COMMAND ====================
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--model", type=str, default="my_model", help="Model name")
    
    # ==================== PREDICT COMMAND ====================
    predict_parser = subparsers.add_parser("predict", help="Make a prediction")
    predict_parser.add_argument("input", type=str, help="Input data for prediction")
    predict_parser.add_argument("--model", type=str, default="best_model.pth", help="Path to model")
    
    # ==================== EVALUATE COMMAND ====================
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    evaluate_parser.add_argument("--dataset", type=str, default="test.csv", help="Dataset to evaluate on")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to the correct function
    if args.command == "train":
        train_model(args.epochs, args.lr, args.model)
    elif args.command == "predict":
        predict(args.input, args.model)
    elif args.command == "evaluate":
        evaluate(args.dataset)
    else:
        parser.print_help()


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    main()


# ==================== HOW TO USE ====================
"""
How to use this CLI tool:

1. Basic help:
   python cli_tool.py --help

2. Train a model:
   python cli_tool.py train --epochs 50 --lr 0.005 --model sentiment_model

3. Make a prediction:
   python cli_tool.py predict "This movie was amazing!" --model best_model.pth

4. Evaluate:
   python cli_tool.py evaluate --dataset test_data.csv
"""