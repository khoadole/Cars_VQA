from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import configparser
from datasets import load_dataset
import pandas as pd
import os

### BOW Vocab
def load_corpus(file_path:str) -> list:
    '''Load corpus from file'''
    # Check if the file exists in the current directory or in the parent directory
    if not os.path.exists(file_path):
        alternative_paths = [
            os.path.join("../", file_path),
            os.path.join("../../", file_path),
            os.path.join(os.path.dirname(__file__), file_path)
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                file_path = path
                break
    with open(file_path, 'r') as f:
        corpus = f.readlines()
    corpus = [q.strip() for q in corpus if q.strip()]
    return corpus

def create_TFIDF(corpus:str, max_features=384, oov_token="<unk>") -> TfidfVectorizer:
    '''Create a BOW model from the corpus'''
    # Add OOV token to the corpus
    corpus_with_oov = corpus + [oov_token]
    vectorizer = TfidfVectorizer(max_features=max_features, binary=True, token_pattern=r"(?u)\b\w+\b|<unk>")
    vectorizer.fit(corpus_with_oov)
    return vectorizer

def vectorize_question_to_TFIDF(vectorizer:TfidfVectorizer, question:str, oov_token="<unk>") -> np.ndarray:
    '''Vector a question using the BOW model'''
    # Preprocess the question
    question = question.lower()
    tokens = re.findall(r"\w+|[^\w\s]", question)
    # Replace OOV tokens
    processed = [
        token if (not token.isalpha() or token in vectorizer.get_feature_names_out()) else oov_token
        for token in tokens
    ]
    question = " ".join(processed)
    # Vectorize the question
    vec = vectorizer.transform([question]).toarray().squeeze()
    return vec

def create_global_dictionaries(train_df, val_df, test_df):
    all_questions = pd.concat([train_df['question'], val_df['question'], test_df['question']])
    
    all_answers = pd.concat([train_df['answer'], val_df['answer'], test_df['answer']]).str.lower()
    
    global_questions_bow = create_TFIDF(all_questions)
    
    unique_answers = all_answers.unique()
    global_answers_to_idx = {answer: idx for idx, answer in enumerate(unique_answers)}
    
    return global_questions_bow, global_answers_to_idx

def create_local_dictionaries(train_df):
    questions_bow = create_TFIDF(train_df['question'])
    answers = train_df['answer'].str.lower()
    unique_answers = answers.unique()
    answers_to_idx = {answer: idx for idx, answer in enumerate(unique_answers)}
    return questions_bow, answers_to_idx

def create_answer(dataframe:pd.DataFrame) -> dict:
    """Load answer mapping from dataset"""
    answer_unique = dataframe['answer'].str.lower().unique()
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_unique)}
    return answer_to_idx

## Test
# Calling for testing
def load_config(config_path:str) -> configparser.ConfigParser:
    """Load configuration from config file"""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def load_answer_mapping(config:configparser.ConfigParser) -> dict:
    """Load answer mapping from dataset"""
    answer_path = config['dataset']['dataset_hf']
    original_cars_df = load_dataset(answer_path)
    cars_df = pd.DataFrame(original_cars_df["train"])
    answer_unique = cars_df.answer.unique()
    answer_to_idx = {answer: i for i, answer in enumerate(answer_unique)}
    return answer_to_idx

def load_question_bow(config:configparser.ConfigParser) -> TfidfVectorizer:
    """Load corpus and create vectorizer"""
    brand_questions_path = config['question path']['question_brand']
    color_questions_path = config['question path']['question_color']
    type_questions_path = config['question path']['question_type']
    
    # Make corpus
    corpus_brand = load_corpus(brand_questions_path)
    corpus_color = load_corpus(color_questions_path)
    corpus_type = load_corpus(type_questions_path)
    corpus = corpus_brand + corpus_color + corpus_type
    
    # Create BOW for corpus
    vectorizer = create_TFIDF(corpus)
    return vectorizer

## Main
if __name__ == "__main__":
    # Question
    vectorizer = load_question_bow(load_config('../../../../config/config.ini'))
    # Test vectorize a question
    vec = vectorize_question_to_TFIDF(vectorizer, "What is the color of this car fuck brand, and car name ?")
    print("vector:", vec)

    ## Answer
    answer = load_answer_mapping(load_config('../../../../config/config.ini'))
    print("Answer mapping:", answer)
    
