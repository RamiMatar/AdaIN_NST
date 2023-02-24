import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageFile

class ContentStyleDataset(torch.utils.data.Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform
        self.content_filenames = [filename for filename in os.listdir(content_dir) if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')]
        self.style_filenames = [filename for filename in os.listdir(style_dir) if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]

    def __len__(self):
        return len(self.content_filenames)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.content_dir, self.content_filenames[idx])
        content_image = Image.open(img_path)
        if content_image.mode != 'RGB':
            content_image = content_image.conert('RGB')
        random_style_idx = torch.randint(low = 0, high = len(self.style_filenames), size=(1,))
        style_path = os.path.join(self.style_dir, self.style_filenames[random_style_idx])
        style_image = Image.open(style_path)
        if style_image.mode != 'RGB':
            style_image = style_image.convert('RGB')
        if self.transform:
            content = self.transform(content_image)
            style = self.transform(style_image)
        return content, style
    

def collate(batch):
    content = []
    style = []
    for i in range(len(batch)):
        content.append(batch[i][0])
        style.append(batch[i][1])
    content = torch.stack(content)
    style = torch.stack(style)
    return torch.stack((content, style))
ImageFile.LOAD_TRUNCATED_IMAGES=True
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)), 
    transforms.ToTensor()])
