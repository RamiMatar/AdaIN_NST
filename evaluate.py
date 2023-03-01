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
from dataset import collate, ContentStyleDataset, loader, ImageFolderDataset, denorm
from net import Model
import os
from collections import OrderedDict

class Eval():
    def __init__(self, model, content_directory, style_directory, output_dir, save_all_style_grids = False, alpha = 1.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.model.training = False
        content_dataset = ImageFolderDataset(content_directory, loader)
        style_dataset = ImageFolderDataset(style_directory, loader)
        self.output_dir = output_dir
        self.content_dataloader = DataLoader(content_dataset, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = collate)
        self.style_dataloader = DataLoader(style_dataset, batch_size = 1, shuffle = False, num_workers = 0, collate_fn = collate)
        self.alpha = alpha
        self.step = 0
    
    def evaluate(self):
        self.model.eval()
        num = len(self.content_dataloader)
        progress_bar = trange(0, num, desc = "Evaluating model on images")
        for content in self.content_dataloader:
            content = content.to(self.device)
            self.evaluate_and_save_styled(content)
            progress_bar.update(self.dataloader.batch_size)

    def evaluate_and_save_styled(self, content):
        styles = []
        for idx, style in enumerate(self.style_dataloader):
            style = style.to(self.device)
            _, _, stylized = self.model((content, style))
            grid = vutils.make_grid([denorm(content.squeeze(), self.device), denorm(style.squeeze(), self.device), denorm(stylized.squeeze(), self.device)], nrow = 3) 
            vutils.save_image(grid, self.output_dir + str(idx).zfill(5) + '_' + str(self.step)+ '.png')
            self.step += 1
            styles.append(denorm(stylized.squeeze(), self.device))
        if self.save_all_style_grids:
            grid = vutils.make_grid(styles, nrow = torch.max(torch.tensor(len(styles)), torch.tensor(6)))
            vutils.save_image(grid, self.output_dir + str(idx).zfill(5) + '_allstyles.png')

    def load_checkpoint(self, PATH):
        state_dict = torch.load(PATH)
        # This is essential if we save the model during DDP training and want to do inference on a single GPU or CPU.
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        print("RESUMING FROM EPOCH ", self.epoch)


def load_from_arguments(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(device, training = False) # skip the loss calculation in inference
    eval = Eval(model, args.content_directory, args.style_directory, args.output_directory, args.save_all_styles_grids)
    return eval

defaults = {
    'content_directory': 'content/',
    'style_directory': 'style/',
    'output_directory': 'output/',
    'model_path': 'checkpoint.pt',
    'save_all_styles_grids': True
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Style Transfer following the AdaIN paper by Huang, et. al.')
    parser.add_argument('content_directory', type=str, help='Directory containing the content images for style transfer')
    parser.add_argument('style_directory', type=str, help='Directory containing the style images for style transfer')
    parser.add_argument('output_directory', type=str, help='where to save the output grids for each image')
    parser.add_argument('model_path', type=str, help='path to the model checkpoint to load for inference')
    parser.add_argument('save_all_styles_grids', type=bool, help='whether to save a grid of all the styles for each content image')
    args = parser.parse_args()
    eval = load_from_arguments(args)
    eval.load_checkpoint(args.model_path)
    eval.evaluate()


