import os
import torch
import pandas as pd
from datasets import load_dataset
import torch.nn as nn
from model import resnet_LSTM
from dataset import data_loader
from preprocess.make_vocab import load_config

CONFIG_PATH = '../../../config/config.ini'
MODEL_NAME = "resnet_LSTM"
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def evaluate():
    config = load_config(CONFIG_PATH)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    
    # log, checkpoint, vocab
    log_base_path = config['log']['log_base_path']
    checkpoint_base_path = config['log']['checkpoint_base_path']
    
    log_dir = os.path.join(project_root, log_base_path, MODEL_NAME, "log")
    log_path = os.path.join(log_dir, "log.txt")
    
    checkpoint_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "checkpoint")
    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")
    
    vocab_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "vocab")
    vocab_path = os.path.join(vocab_dir, "qu_vocab.pth")
    
    dataframe_df = load_dataset(config['dataset']['dataset_hf'])
    train_df = pd.DataFrame(dataframe_df["train"])
    val_df = pd.DataFrame(dataframe_df["validation"])
    test_df = pd.DataFrame(dataframe_df["test"])
    
    dataLoader = data_loader(train_df, val_df, test_df, batch_size=32, shuffle=False, num_workers=4)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    embedding_size = checkpoint['vocab_size']
    num_classes = checkpoint['num_classes']
    
    model = resnet_LSTM(embedding_size=embedding_size, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for image, question, answer in dataLoader["test"]:
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
    evaluate()