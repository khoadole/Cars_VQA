import torch
import torch.nn as nn
import torchvision
    
# EfficientNet LSTM
class EfficientNet_LSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embeddings=None, word_embed=300, lstm_hidden=32):
        super(EfficientNet_LSTM, self).__init__()
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
        
        # LSTM for question
        self.lstm = nn.LSTM(word_embed, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True) # biLSTM
        # self.lstm = nn.LSTM(word_embed, lstm_hidden, num_layers=1, batch_first=True) # LSTM
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 64)
        self.combine_fc2 = nn.Linear(64, num_classes)
        self.tanh = nn.Tanh()
    
    def forward(self, img, question):
        # Image processing
        x = self.features(img)
        img_features = self.classifier(x)
        
        # Question processing
        question_embed = self.word_embedding(question)  
        lstm_out, (hidden, _) = self.lstm(question_embed)  # lstm_out: (batch_size, max_length, lstm_hidden)
        question = hidden[-1]  # (batch_size, lstm_hidden), lấy hidden state cuối
        
        # Combine
        combined = torch.cat((img_features, question), dim=1)  # (batch_size, 64)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out