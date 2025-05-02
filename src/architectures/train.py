import torch
import torch.nn as nn
import torchvision
import pandas as pd 
from datasets import load_dataset
import time
from datetime import datetime
import os
from tqdm import tqdm, trange
import json

from model import *
from dataset import data_loader
from preprocess.make_vocab import load_config

CONFIG_PATH='../../config/config.ini'
EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-4
NUMS_WORKER = 4

def compare_architectures(models_dict, train_loader, val_loader, epochs=10, lr=LR, save_dir='./trained_models'):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # model size in MB
        model_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        ### Training
        print("#"*10 + "TRAINING" + "#"*10)
        start_time = time.time()
        best_acc = 0
        best_model_state = None
        train_losses = []
        val_accs = []
        
        epoch_bar = trange(EPOCHS, desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{EPOCHS}", 
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
            val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{EPOCHS}", 
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
            
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                best_model_state = model.state_dict().copy()
                
                checkpoint_path = f"{save_dir}/{model_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': val_accuracy,
                    'loss': val_accuracy,
                }, checkpoint_path)
                print(f"Saved best checkpoint for {model_name} at epoch {epoch+1} with accuracy {val_accuracy:.2f}%")
                
            # Update the epoch progress bar with summary metrics
            epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", 
                            val_loss=f"{val_loss:.4f}", 
                            train_acc=f"{train_accuracy:.4f}",
                            val_acc=f"{val_accuracy:.4f}")

            print(f'Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        training_time = time.time() - start_time
        
        # inference time
        print("#"*10 + "INFERENCE" + "#"*10)
        inference_start = time.time()
        model.eval()
        with torch.no_grad():
            for images, questions, _ in val_loader:
                images, questions = images.to(device), questions.to(device)
                _ = model(images, questions)
        inference_time = (time.time() - inference_start) / len(val_loader.dataset)
        print(f"Inference time per sample: {inference_time:.4f} seconds")
        model.load_state_dict(best_model_state)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = f"{save_dir}/{model_name}_final_acc{best_acc:.2f}_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': best_acc,
            'model_info': {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': model_size,
                'inference_time': inference_time
            }
        }, final_path)
        
        results.append({
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size,
            'training_time': training_time,
            'average_inference_time': inference_time,
            'best_accuracy': best_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accs,
            'model_path': final_path
        })

    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    return results

def visualize_comparison(results):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'Params (M)': r['total_params'] / 1e6,
        'Size (MB)': r['model_size_mb'],
        'Training Time (s)': r['training_time'],
        'Inference Time (ms)': r['average_inference_time'] * 1000,
        'Accuracy (%)': r['best_accuracy']
    } for r in results])
    
    print(df)
    
    # Plot accuracy
    plt.figure(figsize=(12, 8))
    for r in results:
        plt.plot(range(1, len(r['val_accuracies'])+1), r['val_accuracies'], 
                 label=f"{r['model_name']} (Best: {r['best_accuracy']:.2f}%)")
    
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_comparison.png')
    
    # Plot size vs accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter([r['model_size_mb'] for r in results], 
                [r['best_accuracy'] for r in results],
                s=100)
    
    for r in results:
        plt.annotate(r['model_name'], 
                    (r['model_size_mb'], r['best_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Model Size vs. Accuracy')
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Best Accuracy (%)')
    plt.grid(True)
    plt.savefig('size_vs_accuracy.png')
    
    # Plot inference time vs accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter([r['average_inference_time'] * 1000 for r in results], 
                [r['best_accuracy'] for r in results],
                s=100)
    
    for r in results:
        plt.annotate(r['model_name'], 
                    (r['average_inference_time'] * 1000, r['best_accuracy']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Inference Time vs. Accuracy')
    plt.xlabel('Inference Time per Sample (ms)')
    plt.ylabel('Best Accuracy (%)')
    plt.grid(True)
    plt.savefig('time_vs_accuracy.png')
    
    return df

if __name__ == "__main__":
    config = load_config(CONFIG_PATH)

    # Load dataset
    print("Loading dataset...")
    dataframe_df = load_dataset(config['dataset']['dataset_hf'])
    train_df = pd.DataFrame(dataframe_df["train"])
    val_df = pd.DataFrame(dataframe_df["validation"])
    test_df = pd.DataFrame(dataframe_df["test"])
    dataLoader = data_loader(train_df, val_df, test_df, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMS_WORKER)
    print("Dataset loaded successfully.")

    # Load vocab
    embedding_size = dataLoader["train"].dataset.qu_vocab.vocab_size
    num_classes = dataLoader["train"].dataset.qu_vocab.answer_size

    models_dict = {
        'ResNet_BOW': ResNet_BOW(embedding_size=embedding_size, num_classes=num_classes),
        'VGG_BOW': VGG_BOW(embedding_size=embedding_size, num_classes=num_classes),
        'DenseNet_BOW': DenseNet_BOW(embedding_size=embedding_size, num_classes=num_classes),
        'MobileNet_BOW': MobileNet_BOW(embedding_size=embedding_size, num_classes=num_classes),
        'EfficientNet_BOW': EfficientNet_BOW(embedding_size=embedding_size, num_classes=num_classes),
    }

    # Compare architectures
    results = compare_architectures(models_dict, dataLoader["train"], dataLoader["val"], epochs=EPOCHS, save_dir='./vqa_models')
    # Visualize results
    df = visualize_comparison(results)