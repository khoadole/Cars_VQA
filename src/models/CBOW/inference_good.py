import torch
from preprocess.resize_image import transform_img
from preprocess.make_vocab import load_config
from model import *
from PIL import Image
import requests
from io import BytesIO
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../

# HYPERPARAMETERS
CONFIG_PATH='../../../config/config.ini'
MODEL_NAME = "resnet_CBOW"

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def preprocess_image(image_path):
    """Preprocess the image for the model"""
    transform = transform_img()
    
    try:
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
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def word_to_idx(vocab, word):
    try:
        return list(vocab).index(word)
    except ValueError:
        return list(vocab).index("<unk>")  # Changed to list() for consistency

def preprocess_question(question, vocab):
    """Preprocess the question for the model"""
    question = question.lower().split()
    max_length = 10
    question_indices = [word_to_idx(vocab, token) for token in question]
    if len(question_indices) < max_length:
        question_indices += [word_to_idx(vocab, "<pad>")] * (max_length - len(question_indices))
    else:
        question_indices = question_indices[:max_length]
    question = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    return question

def load_model_and_vocab():
    """Load model weights and vocabulary"""
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = checkpoint['vocab_size']
    num_classes = checkpoint['num_classes']

    # Load embedding
    vocab_data = torch.load(vocab_path, map_location=device, weights_only=False)
    embeddings = vocab_data['embeddings']
    print("Loading weights done.")

    # MODEL    
    model = resnet_CBOW(vocab_size=vocab_size, num_classes=num_classes, embeddings=embeddings, word_embed=300)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab_data, checkpoint

def inference(image_path, question_text, get_top_k=1):
    """
    Perform inference on the given image and question
    
    Args:
        image_path: Path or URL to the image
        question_text: Text of the question to ask
        get_top_k: Number of top predictions to return (default: 1)
    
    Returns:
        For get_top_k=1: The predicted answer string
        For get_top_k>1: List of tuples (answer, confidence) for top k predictions
    """
    # Load model and vocabulary only once
    model, vocab_data, _ = load_model_and_vocab()
    
    # Create inverse mapping for answers
    answer_to_idx = vocab_data['answer_to_idx'] 
    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
    
    # Process the image and question
    image = preprocess_image(image_path)
    if image is None:
        return "Error processing image."
        
    question = preprocess_question(question_text, vocab_data['vocab'])

    # Move tensors to device
    image = image.to(device)
    question = question.to(device)
    
    # Get predictions
    with torch.no_grad():
        output = model(image, question)
        
        if get_top_k == 1:
            # Get the most likely prediction
            _, predicted = torch.max(output.data, 1)
            predicted_answer = idx_to_answer[predicted.item()]
            return predicted_answer
        else:
            # Get top k predictions with probabilities
            probabilities = F.softmax(output, dim=1)
            top_p, top_class = torch.topk(probabilities, get_top_k)
            
            results = []
            for i in range(get_top_k):
                answer = idx_to_answer[top_class[0][i].item()]
                confidence = top_p[0][i].item()
                results.append((answer, confidence))
            
            return results

def interactive_mode():
    """Run an interactive session for VQA"""
    model, vocab_data, _ = load_model_and_vocab()
    
    # Create inverse mapping for answers
    answer_to_idx = vocab_data['answer_to_idx'] 
    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
    
    print("\n===== Visual Question Answering Interactive Mode =====")
    print("Type 'exit' to quit")
    
    while True:
        # Get image path from user
        image_path = input("\nEnter image path or URL (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break
            
        # Get question from user
        question = input("Enter your question about the image: ")
        
        # Get number of answers to show
        try:
            k = int(input("How many answers to show? [1-5]: "))
            k = max(1, min(5, k))  # Limit between 1 and 5
        except:
            k = 1
            
        # Process the inputs
        image = preprocess_image(image_path)
        if image is None:
            print("Error processing image. Please check the path/URL and try again.")
            continue
            
        question_tensor = preprocess_question(question, vocab_data['vocab'])
        
        # Move tensors to device
        image = image.to(device)
        question_tensor = question_tensor.to(device)
        
        # Get predictions
        with torch.no_grad():
            output = model(image, question_tensor)
            probabilities = F.softmax(output, dim=1)
            top_p, top_class = torch.topk(probabilities, k)
            
            print("\n----- Results -----")
            for i in range(k):
                answer = idx_to_answer[top_class[0][i].item()]
                confidence = top_p[0][i].item()
                print(f"{i+1}. {answer} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visual Question Answering Inference')
    parser.add_argument('--image', type=str, help='Path or URL to image')
    parser.add_argument('--question', type=str, help='Question about the image')
    parser.add_argument('--top', type=int, default=1, help='Number of top answers to return')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.image and args.question:
        results = inference(args.image, args.question, args.top)
        
        if isinstance(results, list):  # Multiple results
            print("\nTop predictions:")
            for i, (answer, confidence) in enumerate(results):
                print(f"{i+1}. {answer} (confidence: {confidence:.4f})")
        else:  # Single result
            print(f"\nPredicted Answer: {results}")
    else:
        # Example usage
        image_path = "https://di-uploads-pod10.dealerinspire.com/acuranorthscottsdale/uploads/2018/09/2019-mdx-png.png"
        question = "Đây là chiếc xe brand what?"
        predicted_answer = inference(image_path, question)
        print(f"Predicted Answer: {predicted_answer}")