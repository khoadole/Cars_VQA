import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import mul, cat, tanh, relu

# resnet CBOW
class resnet_CBOW(nn.Module):
    def __init__(self, vocab_size, num_classes, embeddings=None, word_embed=300):
        super(resnet_CBOW, self).__init__()
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 32)
        )
        
        # Word Embedding
        self.word_embedding = nn.Embedding(vocab_size, word_embed)
        if embeddings is not None:
            self.word_embedding.weight = nn.Parameter(torch.tensor(embeddings, dtype=torch.float32))
            self.word_embedding.weight.requires_grad = False  # Không huấn luyện lại embedding
        self.embedding_fc = nn.Linear(word_embed, 32)
        self.tanh = nn.Tanh()
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 64)
        self.combine_fc2 = nn.Linear(64, num_classes)
    
    def forward(self, img, question):
        img_features = self.resnet(img)
        
        question_embed = self.word_embedding(question)  # (batch_size, max_length, word_embed)
        question = question_embed.mean(dim=1)  # (batch_size, word_embed)
        question = self.tanh(self.embedding_fc(question))
        
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out