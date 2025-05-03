from torchvision import transforms

def transform_img():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.224, 0.239, 0.258]),
    ])
    return transform  