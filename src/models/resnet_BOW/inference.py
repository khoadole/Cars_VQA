from preprocess.resize_image import transform_img
from preprocess.make_vocab import load_config, vectorize_question_to_bow
from model import *
from PIL import Image
import requests
from io import BytesIO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../

# HYPERPARAMETERS
CONFIG_PATH='../../../config/config.ini'
MODEL_NAME = "resnet_BOW"

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    """Preprocess the image for the model"""
    transform = transform_img()
    
    if image_path.startswith(('http://', 'https://')):
        # URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        # local file path
        image = Image.open(image_path)

    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def preprocess_question(question, vectorizer):
    """Preprocess the question for the model"""
    question_vec = vectorize_question_to_bow(vectorizer, question)
    question_tensor = torch.tensor(question_vec, dtype=torch.float).unsqueeze(0)  # (1, embedding_size)
    return question_tensor

def inference(image:str, question:str):
    config = load_config(CONFIG_PATH)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")) 

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
    
    # Load weights
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_size = checkpoint['vocab_size']
    num_classes = checkpoint['num_classes']
    vocab_data = torch.load(vocab_path, map_location=device, weights_only=False)
    print("Loading weights done.")

    # MODEL    
    model = resnet_BOW(embedding_size=embedding_size, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Process the image and question
    image = preprocess_image(image)
    question = preprocess_question(question, vocab_data['q_bow'])

    image = image.to(device)
    question = question.to(device)
    with torch.no_grad():
        output = model(image, question)
        _, predicted = torch.max(output.data, 1)

    # Predict
    # idx_to_answer = vocab_data['answer_to_idx']
    answer_to_idx = {v: k for k, v in vocab_data['answer_to_idx'].items()}  # Invert the dictionary
    predicted_answer = answer_to_idx[predicted.item()]
    return predicted_answer

if __name__ == "__main__":
    # image_path = "images/images.jpeg"
    image_path = "https://di-uploads-pod10.dealerinspire.com/acuranorthscottsdale/uploads/2018/09/2019-mdx-png.png"
    question = "Đây là chiếc xe brand what?"
    predicted_answer = inference(image_path, question)
    print(f"Predicted Answer: {predicted_answer}")