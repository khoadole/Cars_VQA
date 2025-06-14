from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess.resize_image import transform_img
from preprocess.make_vocab import (
    create_answer,
    create_global_dictionaries
)

CONFIG_PATH = '../../config/config.ini'

class VocabInfo:
    def __init__(self, vocab, embeddings, ans_dict):
        self.vocab_size = len(vocab)
        self.answer_size = len(ans_dict)
        self.vocab = vocab  # Danh sách từ vựng
        self.embeddings = embeddings  # Ma trận embedding
        self.answer_to_idx = ans_dict
    
    def get_vocab(self):
        return self.vocab
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_answer_to_idx(self):
        return self.answer_to_idx
    
    def word_to_idx(self, word):
        try:
            return list(self.vocab).index(word)
        except ValueError:
            return self.vocab.index("<unk>")
    
    def idx_to_word(self, idx):
        if idx < len(self.vocab):
            return self.vocab[idx]
        return "<unk>"

class Dataset_CBOW(Dataset):
    def __init__(self, dataframe, transform=None, vocab=None, embeddings=None, answers_to_idx=None):
        self.dataframe = dataframe
        self.transform = transform
        self.vocab = vocab
        self.embeddings = embeddings
        self.answers_to_idx = answers_to_idx
        self.qu_vocab = VocabInfo(self.vocab, self.embeddings, self.answers_to_idx)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Image 
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        if self.transform:
            image = self.transform(image)
        
        # Question to idx
        question = row['question'].lower().split()
        max_length = 10
        question_indices = [self.qu_vocab.word_to_idx(token) for token in question]
        if len(question_indices) < max_length:
            question_indices += [self.qu_vocab.word_to_idx("<pad>")] * (max_length - len(question_indices))
        else:
            question_indices = question_indices[:max_length]
        question = torch.tensor(question_indices, dtype=torch.long)
        
        # Answer
        answer = row['answer'].lower()
        if answer in self.answers_to_idx:
            answer_idx = self.answers_to_idx[answer]
        else:
            answer_idx = 0
        return image, question, answer_idx

# def data_loader(train_df, val_df, test_df, embedding=None, answers_to_idx=None, vocabs=None, batch_size=8, shuffle=True, num_workers=4):
#     if embedding is None or answers_to_idx is None or vocabs is None:
#         vocab, embeddings, global_answers_to_idx = create_global_dictionaries(train_df, val_df, test_df)
#     else :
#         global_answers_to_idx = answers_to_idx
#         vocab = vocabs
#         embeddings = embedding

#     # Transform
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.224, 0.239, 0.258]),
#     ])
    
#     vqa_dataset = {
#         'train': Dataset_CBOW(
#             dataframe=train_df,
#             transform=transform,
#             vocab=vocab,
#             embeddings=embeddings,
#             answers_to_idx=global_answers_to_idx),
#         'val': Dataset_CBOW(
#             dataframe=val_df,
#             transform=transform,
#             vocab=vocab,
#             embeddings=embeddings,
#             answers_to_idx=global_answers_to_idx),
#         'test': Dataset_CBOW(
#             dataframe=test_df,
#             transform=transform,
#             vocab=vocab,
#             embeddings=embeddings,
#             answers_to_idx=global_answers_to_idx)
#     }
    
#     data_loader = {
#         key: DataLoader(vqa_dataset[key], 
#                         batch_size=batch_size, 
#                         shuffle=shuffle, 
#                         num_workers=num_workers) 
#         for key in ['train', 'val', 'test']
#     }
#     return data_loader

def data_loader(train_df, val_df, test_df, vocabs=None, batch_size=8, shuffle=True, num_workers=4):
    if vocabs is None:
        vocab, embeddings, global_answers_to_idx = create_global_dictionaries(train_df, val_df, test_df)
    else:
        vocab = vocabs['vocab']  # Lấy vocab từ vocabs
        embeddings = vocabs['embeddings']  # Lấy embeddings từ vocabs
        global_answers_to_idx = vocabs['answer_to_idx'] # Load answer to index dict
    
    vqa_dataset = {
        'train': Dataset_CBOW(
            dataframe=train_df,
            transform=transform_img(),
            vocab=vocab,
            embeddings=embeddings,
            answers_to_idx=global_answers_to_idx),
        'val': Dataset_CBOW(
            dataframe=val_df,
            transform=transform_img(),
            vocab=vocab,
            embeddings=embeddings,
            answers_to_idx=global_answers_to_idx),
        'test': Dataset_CBOW(
            dataframe=test_df,
            transform=transform_img(),
            vocab=vocab,
            embeddings=embeddings,
            answers_to_idx=global_answers_to_idx)
    }
    
    data_loader = {
        key: DataLoader(vqa_dataset[key], 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers) 
        for key in ['train', 'val', 'test']
    }
    return data_loader
