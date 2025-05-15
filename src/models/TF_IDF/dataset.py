from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.resize_image import transform_img
from preprocess.make_vocab import (
	vectorize_question_to_TFIDF,
	create_TFIDF,
	create_answer,
	create_global_dictionaries,
    create_local_dictionaries
)
CONFIG_PATH='../../../config/config.ini'

### Dataset TFIDF global
class Dataset_TFIDF(Dataset):
    def __init__(self, dataframe, transform=None, questions_bow=None, answers_to_idx=None):
        self.dataframe = dataframe
        self.transform = transform
        
        self.questions_bow = questions_bow
        self.answers_to_idx = answers_to_idx
        
        self.qu_vocab = VocabInfo(self.questions_bow, self.answers_to_idx)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # Image 
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        if self.transform:
            image = self.transform(image) # Tensor : (3, 224, 224)
        
        # Question
        question = row['question']
        question = vectorize_question_to_TFIDF(self.questions_bow, question)
        
        # Answer
        answer = row['answer'].lower()
        
        # If answer is not in the dictionary, assign a default index (0)
        if answer in self.answers_to_idx:
            answer_idx = self.answers_to_idx[answer]
        else:
            answer_idx = 0  
            
        return image, question, answer_idx

def data_loader(train_df, val_df, test_df, vocabs=None, batch_size=8, shuffle=True, num_workers=4):
    # global_questions_bow, global_answers_to_idx = create_global_dictionaries(train_df, val_df, test_df)
    if vocabs is None:
        local_questions_bow, local_answers_to_idx = create_local_dictionaries(train_df)
    else:
        local_questions_bow = vocabs['q_bow'] 
        local_answers_to_idx = vocabs['answer_to_idx']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.224, 0.239, 0.258]),
    ])
    
    vqa_dataset = {
        'train': Dataset_TFIDF(
            dataframe=train_df,
            transform=transform,
            questions_bow=local_questions_bow,
            answers_to_idx=local_answers_to_idx),
        'val': Dataset_TFIDF(
            dataframe=val_df,
            transform=transform,
            questions_bow=local_questions_bow,
            answers_to_idx=local_answers_to_idx),
        'test': Dataset_TFIDF(
            dataframe=test_df,
            transform=transform,
            questions_bow=local_questions_bow,
            answers_to_idx=local_answers_to_idx)
    }
    
    data_loader = {
        key : DataLoader(vqa_dataset[key], 
                   batch_size=batch_size, 
                   shuffle=shuffle, 
                   num_workers=num_workers) 
        for key in ['train', 'val', 'test']
    }
    return data_loader

# class VocabInfo:
#     def __init__(self, q_bow, ans_dict):
#         self.vocab_size = len(q_bow.get_feature_names_out())
#         self.answer_size = len(ans_dict)

class VocabInfo:
    def __init__(self, q_bow, ans_dict):
        self.vocab_size = len(q_bow.get_feature_names_out())
        self.answer_size = len(ans_dict)
        self.vocab = q_bow.get_feature_names_out()  # Lưu danh sách từ vựng (dạng mảng các từ)
        self.answer_to_idx = ans_dict  # Lưu từ điển ánh xạ từ câu trả lời sang chỉ số
    
    def get_vocab(self):
        return self.vocab
    
    def get_answer_to_idx(self):
        return self.answer_to_idx
    
    def word_to_idx(self, word):
        try:
            return list(self.vocab).index(word)
        except ValueError:
            return 0  

    def idx_to_word(self, idx):
        if idx < len(self.vocab):
            return self.vocab[idx]
        return "<unk>" 