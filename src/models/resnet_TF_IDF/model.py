import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import mul, cat, tanh, relu

class resnet_TFIDF(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(resnet_TFIDF, self).__init__()
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 32)
        )
        
        # question
        self.ques_fc1 = nn.Linear(embedding_size, 64)
        self.ques_fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        
        # combine
        self.combine_fc1 = nn.Linear(64, 32)
        self.combine_fc2 = nn.Linear(32, num_classes)
    
    def forward(self, img, question):
        # image process
        img_features = self.resnet(img)
        
        # question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))
        question = self.tanh(self.ques_fc2(question))
        
        # combined
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out
    
class EfficientNet_TFIDF(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(EfficientNet_TFIDF, self).__init__()
        # self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        # num_ftrs = self.resnet.fc.in_features
        # self.resnet.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 32)
        # )
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
        
        # question
        self.ques_fc1 = nn.Linear(embedding_size, 64)
        self.ques_fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        
        # combine
        self.combine_fc1 = nn.Linear(64, 32)
        self.combine_fc2 = nn.Linear(32, num_classes)
    
    def forward(self, img, question):
        # image process
        x = self.features(img)
        img_features = self.classifier(x)
        
        # question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))
        question = self.tanh(self.ques_fc2(question))
        
        # combined
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out