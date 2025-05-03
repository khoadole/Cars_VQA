import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import mul, cat, tanh, relu

### Resnet + BOW : split image and question encoder
class ImgEncoder_Resnet(nn.Module):
	def __init__(self):
		super(ImgEncoder_Resnet, self).__init__()
		self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
		num_ftrs = self.resnet.fc.in_features
		self.resnet.fc = nn.Sequential(
			torch.nn.Linear(num_ftrs, 512),
			torch.nn.Tanh(),
			torch.nn.Linear(512, 128),
			torch.nn.Tanh(),
			torch.nn.Linear(128, 32)
		)
	
	def forward(self, image):
		with torch.no_grad():
			img_features = self.resnet(image)
		return img_features

class QuestionEncoder_BOW(nn.Module):
	def __init__(self, embedding_size):
		super(QuestionEncoder_BOW, self).__init__()
		self.embedding_size = embedding_size

		self.fc1 = nn.Linear(self.embedding_size, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 32)
		self.tanh = nn.Tanh()

	def forward(self, question):
		question = question.float()
		question = self.fc1(question)
		question = self.tanh(question)

		question = self.fc2(question)
		question = self.tanh(question)

		out = self.fc3(question)
		return out

class resnetBOW_split(nn.Module):
	def __init__(self, embedding_size, num_classes):
		super(resnetBOW_split, self).__init__()
		self.img_encoder = ImgEncoder_Resnet()
		self.question_encoder = QuestionEncoder_BOW(embedding_size)

		self.tanh = nn.Tanh()
		self.fc1 = nn.Linear(64, 32)
		self.fc2 = nn.Linear(32, num_classes)
	
	def forward(self, img, question):
		img_feature = self.img_encoder(img)               # (batchsize, 32)
		ques_feature = self.question_encoder(question)		# (batchsize, 32)

		# concat
		out = cat((img_feature, ques_feature), 1)   
		out = self.tanh(self.fc1(out))
		out = self.fc2(out)
		return out

# resnetBOW : no split
class resnet_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(resnet_BOW, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
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
	
# resnet + BƠW : change concat => multiply, embedding_size → 128 → 32 (embedding_size → 64 → 32), same resnet
class resnetBOW_mul(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(resnetBOW_mul, self).__init__()
        self.resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 32)
        )
        
        # Question
        self.ques_fc1 = nn.Linear(embedding_size, 128)  # Tăng từ 64 lên 128
        self.ques_fc2 = nn.Linear(128, 32)
        self.tanh = nn.Tanh()
        
        # Combine
        self.combine_fc1 = nn.Linear(32, 64)
        self.combine_fc2 = nn.Linear(64, num_classes)
    
    def forward(self, img, question):
        # Image process
        img_features = self.resnet(img)  # (batch_size, 32)
        
        # Question process
        question = question.float()
        question = self.tanh(self.ques_fc1(question))  # (batch_size, 128)
        question = self.tanh(self.ques_fc2(question))  # (batch_size, 32)
        
        # Combined
        combined = img_features * question  # (batch_size, 32) * (batch_size, 32) = (batch_size, 32)
        out = self.tanh(self.combine_fc1(combined))  # (batch_size, 64)
        out = self.combine_fc2(out)  # (batch_size, num_classes)
        return out
	
# resnet + BOW : change resnet output layer
class resnetBOW_outputlayer_cat(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(resnetBOW_outputlayer_cat, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.Tanh(),
			nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
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
	
# EfficientNet + BOW
class EfficientNet_BOW(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(EfficientNet_BOW, self).__init__()
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