import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

imsize = 512 if torch.cuda.is_available() else 128

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

loader = transforms.Compose([transforms.Resize(imsize),
                            transforms.ToTensor(),
                            normalize])

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

class ContentStyleDataset(torch.utils.data.Dataset):
    def __init__(self, content_dir, style_dir, transform=loader):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform
        self.content_filenames = [content_dir + "/" + filename for filename in os.listdir(content_dir) if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')]
        #self.content_filenames.extend([albums_dir + "/" + filename for filename in os.listdir(albums_dir) if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('jpeg')])
        self.style_filenames = [filename for filename in os.listdir(style_dir) if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]
        self.content_file_map = {index: filename for index, filename in enumerate(self.content_filenames)}
        self.style_content_map  = {index: filename for index, filename in enumerate(self.style_filenames)}
        print(len(self.content_filenames))

    def __len__(self):
        return len(self.content_filenames)
        
    def __getitem__(self, idx):
        content_image = Image.open(self.content_filenames[idx])
        if content_image.mode != 'RGB':
            content_image = content_image.convert('RGB')
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

