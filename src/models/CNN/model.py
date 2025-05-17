import torch
import torch.nn as nn
import torchvision

class EfficientNet_CNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embeddings=None, word_embed=300, cnn_output=32):
        super(EfficientNet_CNN, self).__init__()
        efficientnet = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = efficientnet.features
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(True),
            nn.Linear(512, 32)
        )
        
        # Word Embedding for question
        self.word_embedding = nn.Embedding(vocab_size, word_embed)
        if embeddings is not None:
            self.word_embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))
            self.word_embedding.weight.requires_grad = False
        
        # CNN for text processing
        self.conv1 = nn.Conv1d(word_embed, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, cnn_output, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 64)
        self.combine_fc2 = nn.Linear(64, num_classes)
        self.tanh = nn.Tanh()
    
    def forward(self, img, question):
        # Image processing
        x = self.features(img)
        img_features = self.classifier(x)
        
        # Question processing with CNN
        question_embed = self.word_embedding(question)  # (batch_size, seq_len, embed_dim)
        question_embed = question_embed.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        
        # Apply CNN layers
        question = self.relu(self.conv1(question_embed))
        question = self.pool(question)
        question = self.dropout(question)
        
        question = self.relu(self.conv2(question))
        question = self.global_pool(question)  # (batch_size, cnn_output, 1)
        question = question.squeeze(-1)  # (batch_size, cnn_output)
        
        # Combine
        combined = torch.cat((img_features, question), dim=1)  # (batch_size, 64)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out