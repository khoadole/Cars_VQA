import torch 
import torch.nn as nn
from torch.optim import Adam
from datasets import load_dataset
import pandas as pd

import time
from datetime import timedelta
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
MODEL_NAME = "resnet_TFIDF"

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def train():
    start_time = time.time()
    
    config = load_config(CONFIG_PATH)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) 

    # Config log save path
    log_base_path = config['log']['log_base_path']
    log_dir = os.path.join(project_root, log_base_path, MODEL_NAME, "log")
    log_path = os.path.join(log_dir, "log.txt")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Config checkpoint save path
    checkpoint_base_path = config['log']['checkpoint_base_path']
    checkpoint_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "checkpoint")
    checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Config vocab save path
    vocab_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "vocab")
    vocab_path = os.path.join(vocab_dir, "qu_vocab.pth")
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    
    # Load Dataset
    print("Loading dataset...")
    dataframe_df = load_dataset(config['dataset']['dataset_hf'])
    train_df = pd.DataFrame(dataframe_df["train"])
    val_df = pd.DataFrame(dataframe_df["validation"])
    test_df = pd.DataFrame(dataframe_df["test"])
    dataLoader = data_loader(train_df, val_df, test_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMS_WORKER)
    print("Dataset loaded successfully.")

    # Add data for save
    embedding_size = dataLoader["train"].dataset.qu_vocab.vocab_size
    num_classes = dataLoader["train"].dataset.qu_vocab.answer_size

    # Save vocab
    qu_vocab = dataLoader["train"].dataset.qu_vocab
    torch.save({
        'q_bow': dataLoader["train"].dataset.questions_bow,  
        'answer_to_idx': qu_vocab.answer_to_idx  
    }, vocab_path)
    print(f"Vocabulary saved to {vocab_path}")

    # MODEL
    model = resnet_TFIDF(embedding_size=embedding_size, num_classes=num_classes)
    model = model.to(device)

    # Parameters
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    print(f"Model initialized and moved to : {device}")

    # Default parameters model for log
    with open(log_path, 'w') as log_file:
        log_file.write(f"Model Name: {MODEL_NAME}\n")
        log_file.write(f"Hyperparameters:\n")
        log_file.write(f"  Learning Rate: {LR}\n")
        log_file.write(f"  Optimizer: Adam\n")
        log_file.write(f"  Batch Size: {BATCH_SIZE}\n")
        log_file.write(f"  Epochs: {EPOCHS}\n")
        log_file.write("\nTraining Log:\n")
        log_file.write("=" * 50 + "\n")

    ### Training
    print("#"*10 + "TRAINING" + "#"*10)

    epoch_bar = trange(EPOCHS, desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(dataLoader["train"], desc=f"Train Epoch {epoch+1}/{EPOCHS}", 
                         unit="batch", leave=False)
        for image, question, answer in train_bar:
            image = image.to(device)
            question = question.to(device)
            answer = answer.to(device)
            
            output = model(image, question)
            loss = criterion(output, answer)
            
            _, predicted = torch.max(output.data, 1)
            train_total += answer.size(0)
            train_correct += (predicted == answer).sum().item()
            
            train_loss += loss.item()
            
            # Update train progress bar with current loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}", 
                                 accuracy=f"{(predicted == answer).sum().item() / answer.size(0):.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(dataLoader["train"])
        train_accuracy = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Create a tqdm progress bar for the validation batches
        val_bar = tqdm(dataLoader["val"], desc=f"Val Epoch {epoch+1}/{EPOCHS}", 
                       unit="batch", leave=False)
        with torch.no_grad():
            for image, question, answer in val_bar:
                image = image.to(device)
                question = question.to(device)
                answer = answer.to(device)
                
                output = model(image, question)
                loss = criterion(output, answer)
                
                _, predicted = torch.max(output.data, 1)
                val_total += answer.size(0)
                val_correct += (predicted == answer).sum().item()
                
                val_loss += loss.item()
                
                # Update val progress bar with current loss
                val_bar.set_postfix(loss=f"{loss.item():.4f}", 
                                   accuracy=f"{(predicted == answer).sum().item() / answer.size(0):.4f}")
        
        val_loss /= len(dataLoader["val"])
        val_accuracy = val_correct / val_total
        
        # Update the epoch progress bar with summary metrics
        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", 
                            val_loss=f"{val_loss:.4f}", 
                            train_acc=f"{train_accuracy:.4f}",
                            val_acc=f"{val_accuracy:.4f}")
        
        print(f'Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Log results
        with open(log_path, 'a') as log_file:
            log_file.write(f'Epoch {epoch+1}/{EPOCHS}:\n')
            log_file.write(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n')
            log_file.write(f'  Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}\n')
            log_file.write("-" * 50 + "\n")
        # Save checkpoint
        checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'vocab_size': embedding_size,
                'num_classes': num_classes
            }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
        
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    # Save training time to log
    print(f"Total Training Time: {total_time_str}")
    with open(log_path, 'a') as log_file:
        log_file.write("\nTraining Summary:\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Total Training Time: {total_time_str}\n")
        log_file.write("=" * 50 + "\n")
        
    print("-" * 50)
    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Log saved to {log_path}")

if __name__ == '__main__':
    train()