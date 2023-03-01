import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

imsize = 512 if torch.cuda.is_available() else 128

# load the transforms for the vgg model
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

loader = transforms.Compose([transforms.Resize([imsize, imsize]),
                            transforms.ToTensor(),
                            normalize])

def get_all_jpg_files(path):
    '''Get all the jpg files in the path recursively, useful for labelled datasets'''
    jpg_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def denorm(tensor, device):
    '''Denormalize the tensor to be in the range [0, 1] to invert the transforms for vgg model,
    this is necessary before the output tensor can be converted to the desired output image'''
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

class ContentStyleDataset(torch.utils.data.Dataset):
    '''Dataset class for the content and style images, getitem returns a random style image as a pair to the given content image.
    '''
    def __init__(self, content_dir, style_dir, transform=loader):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform
        self.content_filenames = get_all_jpg_files(content_dir)
        self.style_filenames = get_all_jpg_files(style_dir)

    def __len__(self):
        return len(self.content_filenames)
        
    def __getitem__(self, idx):
        content_image = Image.open(self.content_filenames[idx])
        if content_image.mode != 'RGB':
            content_image = content_image.convert('RGB')
        random_style_idx = torch.randint(low = 0, high = len(self.style_filenames), size=(1,))
        style_image = Image.open(self.style_filenames[random_style_idx])
        if style_image.mode != 'RGB':
            style_image = style_image.convert('RGB')
        if self.transform:
            content = self.transform(content_image)
            style = self.transform(style_image)
        return content, style
    
class ImageFolderDataset(torch.utils.data.Dataset):
    '''Dataset for one folder representing either content or style images.'''
    def __init__(self, path, transform=loader):
        self.path = path
        self.transform = transform
        self.filenames = get_all_jpg_files(path)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    

def collate(batch):
    '''Collate function takes the batch as a list of content style pairs and turns it into a tensor of shape (2, batch_size, 3, imsize, imsize)'''
    content = []
    style = []
    for i in range(len(batch)):
        content.append(batch[i][0])
        style.append(batch[i][1])
    content = torch.stack(content)
    style = torch.stack(style)
    return torch.stack((content, style))

