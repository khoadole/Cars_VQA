from .preprocess.resize_image import transform_img
from .preprocess.make_vocab import load_config
from PIL import Image
import requests
from io import BytesIO
import torch
import numpy as np
import onnxruntime as ort

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # ../

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))  # Add app directory
sys.path.append(os.path.join(project_root, 'app', 'model'))  # Add model directory

# HYPERPARAMETERS
CONFIG_PATH='../config/config.ini'
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/config.ini')
MODEL_NAME = "EfficientNet_LSTM"

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

def word_to_idx(vocab, word):
    try:
        return list(vocab).index(word)
    except ValueError:
        return vocab.index("<unk>")

def preprocess_question(question, vocab):
    """Preprocess the question for the model"""
    question = question.lower().split()
    max_length = 10
    question_indices = [word_to_idx(vocab, token) for token in question]
    if len(question_indices) < max_length:
        question_indices += [word_to_idx(vocab, "<pad>")] * (max_length - len(question_indices))
    else:
        question_indices = question_indices[:max_length]
    question = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
    return question

class ONNXInferenceEngine:
    """ONNX Inference Engine for Visual Question Answering"""
    
    def __init__(self, onnx_path, vocab_path):
        """
        Initialize ONNX inference engine
        
        Args:
            onnx_path: Path to ONNX model file
            vocab_path: Path to vocabulary file
        """
        self.onnx_path = onnx_path
        self.vocab_path = vocab_path
        self.session = None
        self.vocab_data = None
        self.answer_to_idx = None
        self._load_model()
        self._load_vocab()
    
    def _load_model(self):
        """Load ONNX model"""
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found at: {self.onnx_path}")
        
        print("Loading ONNX model...")
        try:
            # Set providers (GPU if available, otherwise CPU)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            
            # Get input and output info
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"ONNX model loaded successfully using: {self.session.get_providers()}")
            print(f"Input names: {self.input_names}")
            print(f"Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _load_vocab(self):
        """Load vocabulary data"""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {self.vocab_path}")
        
        print("Loading vocabulary...")
        self.vocab_data = torch.load(self.vocab_path, map_location='cpu', weights_only=False)
        
        # Create answer index mapping
        self.answer_to_idx = {v: k for k, v in self.vocab_data['answer_to_idx'].items()}
        print("Vocabulary loaded successfully.")
    
    def predict(self, image_path, question):
        """
        Make prediction using ONNX model
        
        Args:
            image_path: Path to image or URL
            question: Question string
            
        Returns:
            predicted_answer: String answer
        """
        # Preprocess inputs
        image = preprocess_image(image_path)
        question_tensor = preprocess_question(question, self.vocab_data['vocab'])
        
        # Convert to numpy arrays for ONNX
        image_np = image.numpy()
        question_np = question_tensor.numpy()
        
        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            self.input_names[0]: image_np,      # 'image'
            self.input_names[1]: question_np    # 'question'
        }
        
        # Run inference
        try:
            ort_outputs = self.session.run(self.output_names, ort_inputs)
            output = ort_outputs[0]  # Get the first output
            
            # Get prediction
            predicted_idx = np.argmax(output, axis=1)[0]
            predicted_answer = self.answer_to_idx[predicted_idx]
            
            return predicted_answer
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
    
    def predict_with_confidence(self, image_path, question):
        """
        Make prediction with confidence score
        
        Args:
            image_path: Path to image or URL
            question: Question string
            
        Returns:
            tuple: (predicted_answer, confidence_score)
        """
        # Preprocess inputs
        image = preprocess_image(image_path)
        question_tensor = preprocess_question(question, self.vocab_data['vocab'])
        
        # Convert to numpy arrays for ONNX
        image_np = image.numpy()
        question_np = question_tensor.numpy()
        
        # Prepare inputs for ONNX Runtime
        ort_inputs = {
            self.input_names[0]: image_np,
            self.input_names[1]: question_np
        }
        
        # Run inference
        try:
            ort_outputs = self.session.run(self.output_names, ort_inputs)
            output = ort_outputs[0]  # Get logits
            
            # Apply softmax to get probabilities
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            
            # Get prediction and confidence
            predicted_idx = np.argmax(probabilities, axis=1)[0]
            confidence = probabilities[0][predicted_idx]
            predicted_answer = self.answer_to_idx[predicted_idx]
            
            return predicted_answer, float(confidence)
            
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

def inference(image_path: str, question: str, return_confidence: bool = False):
    """
    Main inference function for backward compatibility
    
    Args:
        image_path: Path to image or URL
        question: Question string
        return_confidence: Whether to return confidence score
        
    Returns:
        predicted_answer or (predicted_answer, confidence)
    """
    config = load_config(CONFIG_PATH)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(current_dir, '..')

    project_root = os.path.abspath(onnx_path)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    # Config paths
    # checkpoint_base_path = config['log']['checkpoint_base_path']
    checkpoint_base_path = "models"
    checkpoint_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "checkpoint")
    onnx_path = os.path.join(checkpoint_dir, "model.onnx")
    
    vocab_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "vocab")
    vocab_path = os.path.join(vocab_dir, "qu_vocab.pth")

    # Check if ONNX model exists
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
    
    # Create inference engine
    engine = ONNXInferenceEngine(onnx_path, vocab_path)
    
    # Make prediction
    if return_confidence:
        return engine.predict_with_confidence(image_path, question)
    else:
        return engine.predict(image_path, question)

def batch_inference(image_paths, questions):
    """
    Batch inference for multiple image-question pairs
    
    Args:
        image_paths: List of image paths/URLs
        questions: List of question strings
        
    Returns:
        List of predicted answers
    """
    if len(image_paths) != len(questions):
        raise ValueError("Number of images and questions must match")
    
    config = load_config(CONFIG_PATH)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Config paths
    checkpoint_base_path = "models"
    checkpoint_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "checkpoint")
    onnx_path = os.path.join(checkpoint_dir, "model.onnx")
    
    vocab_dir = os.path.join(project_root, checkpoint_base_path, MODEL_NAME, "vocab")
    vocab_path = os.path.join(vocab_dir, "qu_vocab.pth")

    # Create inference engine
    engine = ONNXInferenceEngine(onnx_path, vocab_path)
    
    # Process batch
    results = []
    for img_path, question in zip(image_paths, questions):
        try:
            answer = engine.predict(img_path, question)
            results.append(answer)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append("ERROR")
    
    return results