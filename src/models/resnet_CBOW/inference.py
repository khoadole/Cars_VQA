import torch
from torchvision import transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import resnetBOW
from preprocess.make_vocab import load_config, vectorize_question_to_TFIDF  # Thêm import

CONFIG_PATH = '../../config/config.ini'

# Thiết bị
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def preprocess_question(question, vectorizer):
    # Sử dụng vectorize_question_to_TFIDF để vector hóa câu hỏi
    question_vec = vectorize_question_to_TFIDF(vectorizer, question)
    question_tensor = torch.tensor(question_vec, dtype=torch.float).unsqueeze(0)  # (1, embedding_size)
    return question_tensor

def inference(image_path, question, checkpoint_path, vocab_path):
    config = load_config(CONFIG_PATH)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_size = checkpoint['embedding_size']
    num_classes = checkpoint['num_classes']
    
    # Tải từ điển từ vựng và vectorizer
    vocab_data = torch.load(vocab_path, map_location=device)
    idx_to_answer = vocab_data['idx_to_answer']
    
    # Tái tạo vectorizer từ vocab
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(vocabulary=vocab_data['vocab'], binary=True, token_pattern=r"(?u)\b\w+\b|<unk>")
    
    model = resnetBOW(embedding_size=embedding_size, num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    image = preprocess_image(image_path)
    question = preprocess_question(question, vectorizer)
    
    image = image.to(device)
    question = question.to(device)
    
    with torch.no_grad():
        output = model(image, question)
        _, predicted = torch.max(output.data, 1)
    
    predicted_answer = idx_to_answer[predicted.item()]
    return predicted_answer

if __name__ == "__main__":
    MODEL_NAME = "resnetBOW"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    checkpoint_path = os.path.join(project_root, "models", MODEL_NAME, "checkpoint", "model_checkpoint.pth")
    vocab_path = os.path.join(project_root, "models", MODEL_NAME, "vocab", "qu_vocab.pth")
    
    image_path = "path/to/your/image.jpg"
    question = "what is the color of the car"
    
    predicted_answer = inference(image_path, question, checkpoint_path, vocab_path)
    print(f"Predicted Answer: {predicted_answer}")