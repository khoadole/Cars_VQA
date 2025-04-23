import torch 
import pandas as pd
from datasets import load_dataset
from torch.optim import Adam
import torch.nn as nn
from torchvision import transforms, models

from tqdm import tqdm, trange

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../

from model import *
from dataset import data_loader
from preprocess.make_vocab import load_config

# HYPERPARAMETERS
CONFIG_PATH='../../../config/config.ini'
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
NUMS_WORKER = 4
MODEL_NAME = "resnet_CBOW"

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def test():
    config = load_config(CONFIG_PATH)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    # Config log save path
    log_base_path = config['log']['log_base_path']

    log_dir = os.path.join(project_root, log_base_path, MODEL_NAME, "log")
    log_path = os.path.join(log_dir, "log.txt")

    # Config checkpoint save path
    checkpoint_base_path = config['log']['checkpoint_base_path']
    checkpoint_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "checkpoint")
    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")

    # Config vocab save path
    vocab_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "vocab")
    vocab_path = os.path.join(vocab_dir, "qu_vocab.pth")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = checkpoint['vocab_size']
    num_classes = checkpoint['num_classes']

    # Load embedding
    vocab_data = torch.load(vocab_path, map_location=device, weights_only=False)
    embeddings = vocab_data['embeddings']

    # Load Dataframe
    dataframe_df = load_dataset(config['dataset']['dataset_hf'])
    train_df = pd.DataFrame(dataframe_df["train"])
    val_df = pd.DataFrame(dataframe_df["validation"])
    test_df = pd.DataFrame(dataframe_df["test"])
    dataLoader = data_loader(train_df, val_df, test_df, vocabs=vocab_data, batch_size=32, shuffle=False, num_workers=4)

    # MODEL
    model = resnet_CBOW(vocab_size=vocab_size, num_classes=num_classes, embeddings=embeddings, word_embed=300)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Parameter
    criterion = nn.CrossEntropyLoss()

    # Testing
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for image, question, answer in tqdm(dataLoader["test"], desc="Testing", unit="batch"):
            image = image.to(device)
            question = question.to(device)
            answer = answer.to(device)
            
            output = model(image, question)
            loss = criterion(output, answer)
            
            test_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            test_total += answer.size(0)
            test_correct += (predicted == answer).sum().item()
    
    test_loss /= len(dataLoader["test"])
    test_accuracy = test_correct / test_total
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    with open(log_path, 'a') as log_file:
        log_file.write("\nTest Evaluation:\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Test Loss: {test_loss:.4f}\n")
        log_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        log_file.write("=" * 50 + "\n")

if __name__ == "__main__":
    test()