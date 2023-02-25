import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageFile
from dataset import collate, ContentStyleDataset, loader
from net import Model
import os

class Eval():
    def __init__(self, model, dataloader, directory, alpha = 1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.alpha = alpha
        self.step = 0

    def eval(self, num_epochs, load = None):
        epoch = self.load_model() if load != None else 0
        with trange(epoch, num_epochs, desc="All epochs") as epochs:
            for epoch in epochs:
                self.run_epoch(epoch)
    
    def evaluate(self, num):
        self.model.eval()
        progress_bar = trange(0, num, desc = "Evaluating model on images")
        for data in self.dataloader:
            if progress_bar.n >= num:
                break
            data = data.to(self.device)
            self.evaluation_step(data, progress_bar.n)
            progress_bar.update(self.dataloader.batch_size)

    def evaluation_step(self, data, start, ext = '.png'):
        batch_size = self.dataloader.batch_size
        content, style = data
        _, _, stylized = self.model(data)
        content = torch.chunk(content, batch_size, dim = 0)
        style = torch.chunk(style, batch_size, dim = 0)
        stylized = torch.chunk(stylized, batch_size, dim = 0)
        print(content[0].shape, style[0].shape, stylized[0].shape)
        for i in range(batch_size):
            current = start + i
            vutils.save_image([content[i].squeeze(), style[i].squeeze(), stylized[i].squeeze()], str(current) + ext, nrow = 3) 

    def load_checkpoint(self):
        PATH = "checkpoint.pt"
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        print("RESUMING FROM EPOCH ", self.epoch)


def load_train_objects(rank):
    model = Model(rank)
    dataset = ContentStyleDataset('album_covers_512/', 'images/', transform = loader)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, num_workers = 4)
    return model, dataset, dataloader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Style Transfer following the AdaIN paper by Huang, et. al.')
    parser.add_argument('total_images', type=int, help='Total images to evaluate and save the model')
    parser.add_argument('directory', type=int, help='where to save the output grids for each image')
    args = parser.parse_args() 
    model, dataset, dataloader = load_train_objects(0)

