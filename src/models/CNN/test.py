import torch 
import pandas as pd
from datasets import load_dataset
from torch.optim import Adam
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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
MODEL_NAME = "EfficientNet_CNN"

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

    # Load weights
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = checkpoint['vocab_size']
    num_classes = checkpoint['num_classes']

    # Load embedding
    vocab_data = torch.load(vocab_path, map_location=device, weights_only=False)
    embeddings = vocab_data['embeddings']
    print("Loading weights done.")

    # Load Dataset
    print("Loading dataset...")
    dataframe_df = load_dataset(config['dataset']['dataset_hf'])
    train_df = pd.DataFrame(dataframe_df["train"])
    val_df = pd.DataFrame(dataframe_df["validation"])
    test_df = pd.DataFrame(dataframe_df["test"])
    dataLoader = data_loader(train_df, val_df, test_df, vocabs=vocab_data, batch_size=32, shuffle=False, num_workers=4)
    print("Loading dataset done.")

    # MODEL
    model = EfficientNet_CNN(vocab_size=vocab_size, num_classes=num_classes, embeddings=embeddings, word_embed=300)
    
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Parameter
    criterion = nn.CrossEntropyLoss()

    # Testing
    print("#"*10 + "TESTING" + "#"*10)
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
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
    
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(answer.cpu().numpy())
    test_loss /= len(dataLoader["test"])
    test_accuracy = test_correct / test_total

    # precision, recall, F1 score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # confusion matrix
    # conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    # print("Confusion Matrix:")
    # print(conf_matrix)

    #  log
    with open(log_path, 'a') as log_file:
        log_file.write("\nTest Evaluation:\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Test Loss: {test_loss:.4f}\n")
        log_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        log_file.write(f"Precision (weighted): {precision:.4f}\n")
        log_file.write(f"Recall (weighted): {recall:.4f}\n")
        log_file.write(f"F1 Score (weighted): {f1:.4f}\n")
        # log_file.write("Confusion Matrix:\n")
        # log_file.write(str(conf_matrix) + "\n")
        log_file.write("=" * 50 + "\n")
    print(f"Save test log to {log_path}")

if __name__ == "__main__":
    test()