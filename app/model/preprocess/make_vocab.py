import configparser
from datasets import load_dataset
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np

# Hàm hỗ trợ để tạo dữ liệu huấn luyện CBOW
def build_vocab_and_context(corpus, window_size=2):
    vocab = set()
    for sentence in corpus:
        tokens = sentence.lower().split()
        vocab.update(tokens)
    vocab = list(vocab)
    vocab.extend(["<unk>", "<pad>"])
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    # Tạo dữ liệu huấn luyện CBOW: (context, target)
    data = []
    for sentence in corpus:
        tokens = sentence.lower().split()
        for i in range(len(tokens)):
            target = tokens[i]
            context = []
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if j != i:
                    context.append(tokens[j])
            # Đảm bảo context có độ dài cố định
            while len(context) < 2 * window_size:
                context.append("<pad>")
            data.append((context, target))
    return vocab, word_to_idx, idx_to_word, data

# Mô hình CBOW
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embedded = self.embeddings(context)  # (batch_size, context_size, embedding_dim)
        embedded = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        out = self.linear(embedded)  # (batch_size, vocab_size)
        return out

def train_cbow(corpus, embedding_dim=300, window_size=2, epochs=100):
    vocab, word_to_idx, idx_to_word, data = build_vocab_and_context(corpus, window_size)
    vocab_size = len(vocab)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = CBOW(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # tensor
    contexts = []
    targets = []
    for context, target in data:
        context_indices = [word_to_idx[word] for word in context]
        target_idx = word_to_idx[target]
        contexts.append(context_indices)
        targets.append(target_idx)
    
    contexts = torch.tensor(contexts, dtype=torch.long).to(device)
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    
    for epoch in range(epochs):
        model.zero_grad()
        output = model(contexts)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # Lấy word embedding
    embeddings = model.embeddings.weight.detach().cpu().numpy()
    return vocab, embeddings

def create_global_dictionaries(train_df, val_df, test_df, embedding_dim=300):
    all_questions = pd.concat([train_df['question'], val_df['question'], test_df['question']])
    all_questions = all_questions.str.lower()
    
    # Huấn luyện CBOW
    vocab, embeddings = train_cbow(all_questions, embedding_dim=embedding_dim)
    
    # Tạo từ điển ánh xạ câu trả lời
    all_answers = pd.concat([train_df['answer'], val_df['answer'], test_df['answer']]).str.lower()
    unique_answers = all_answers.unique()
    global_answers_to_idx = {answer: idx for idx, answer in enumerate(unique_answers)}
    
    return vocab, embeddings, global_answers_to_idx

def create_answer(dataframe: pd.DataFrame) -> dict:
    answer_unique = dataframe['answer'].str.lower().unique()
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_unique)}
    return answer_to_idx

def load_config(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def load_answer_mapping(config: configparser.ConfigParser) -> dict:
    answer_path = config['dataset']['dataset_hf']
    original_cars_df = load_dataset(answer_path)
    cars_df = pd.DataFrame(original_cars_df["train"])
    answer_unique = cars_df.answer.str.lower().unique()
    answer_to_idx = {answer: i for i, answer in enumerate(answer_unique)}
    return answer_to_idx

if __name__ == "__main__":
    config = load_config('../../../../config/config.ini')
    answer = load_answer_mapping(config)
    print("Answer mapping:", answer)