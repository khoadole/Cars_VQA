import torch
import torch.nn as nn
import torchvision

class ResNet_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(ResNet_BOW, self).__init__()
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

class VGG_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(VGG_BOW, self).__init__()
        vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1")
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 32)
        )
        
        # Question processing
        self.ques_fc1 = nn.Linear(embedding_size, 64)
        self.ques_fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 32)
        self.combine_fc2 = nn.Linear(32, num_classes)
    
    def forward(self, img, question):
        # Image process
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        img_features = self.classifier(x)
        
        # Question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))
        question = self.tanh(self.ques_fc2(question))
        
        # Combined
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out

class MobileNet_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(MobileNet_BOW, self).__init__()
        mobilenet = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(True),
            nn.Linear(512, 32)
        )
        
        # Question processing
        self.ques_fc1 = nn.Linear(embedding_size, 64)
        self.ques_fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 32)
        self.combine_fc2 = nn.Linear(32, num_classes)
    
    def forward(self, img, question):
        # Image process
        x = self.features(img)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        img_features = self.classifier(x)
        
        # Question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))
        question = self.tanh(self.ques_fc2(question))
        
        # Combined
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out

class EfficientNet_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(EfficientNet_BOW, self).__init__()
        efficientnet = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = efficientnet.features
        
        # EfficientNet-B0 có 1280 đặc trưng đầu ra
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(True),
            nn.Linear(512, 32)
        )
        
        # Question processing
        self.ques_fc1 = nn.Linear(embedding_size, 64)
        self.ques_fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 32)
        self.combine_fc2 = nn.Linear(32, num_classes)
    
    def forward(self, img, question):
        # Image process
        x = self.features(img)
        img_features = self.classifier(x)
        
        # Question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))
        question = self.tanh(self.ques_fc2(question))
        
        # Combined
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out

class DenseNet_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(DenseNet_BOW, self).__init__()
        densenet = torchvision.models.densenet121(weights="IMAGENET1K_V1")
        self.features = densenet.features
        
        # DenseNet121 có 1024 đặc trưng đầu ra
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 32)
        )
        
        # Question processing
        self.ques_fc1 = nn.Linear(embedding_size, 64)
        self.ques_fc2 = nn.Linear(64, 32)
        self.tanh = nn.Tanh()
        
        # Combine
        self.combine_fc1 = nn.Linear(64, 32)
        self.combine_fc2 = nn.Linear(32, num_classes)
    
    def forward(self, img, question):
        # Image process
        x = self.features(img)
        img_features = self.classifier(x)
        
        # Question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))
        question = self.tanh(self.ques_fc2(question))
        
        # Combined
        combined = torch.cat((img_features, question), 1)
        out = self.tanh(self.combine_fc1(combined))
        out = self.combine_fc2(out)
        return out

