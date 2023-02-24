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
class Trainer():
    def __init__(self, model, dataloader, learning_rate, save_every, writer, optimizer = "adam", decay_lr = None, alpha = 1.0, k = 10.0):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.lr = learning_rate
        self.save_every = save_every
        self.writer = writer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr = self.lr)
        self.decay_lr = decay_lr
        self.alpha = alpha
        self.k = k

    def train(self, num_epochs, load = None):
        epoch = self.load_model() if load != None else 0
        with trange(epoch, num_epochs, desc="All epochs") as epochs:
            for epoch in epochs:
                self.run_epoch(epoch)

    def logs(self, step, content, style, stylized, style_loss, content_loss):
       # if step == 0:
            #self.writer.add_graph(self.model, (torch.randn(self.dataloader.batch_size, 3, 128, 128), torch.randn(self.dataloader.batch_size, 3, 128, 128)))
        self.writer.add_scalar("Style Loss", style_loss, step)
        self.writer.add_scalar("Content Loss", content_loss, step)
        self.writer.add_scalar("Total Loss", style_loss + content_loss, step)
        self.writer.add_image("Content Images", content[0], step)
        self.writer.add_image("Style Images", style[0], step)
        self.writer.add_image("Stylized Output", stylized[0], step)

    def run_epoch(self, epoch_num):
        num_batches = len(self.dataloader)
        progress_bar = trange(num_batches, desc = "epoch" + str(epoch_num))
        for batch_num, data in enumerate(self.dataloader):
            data = data.to(self.device)
            content_loss, style_loss = self.training_step(epoch_num, batch_num, data)
            progress_bar.update(1)
            progress_bar.set_postfix(content_loss = content_loss.detach().cpu().numpy(), style_loss = style_loss.detach().cpu().numpy(), total_loss = (content_loss + style_loss).detach().cpu().numpy())
            if batch_num % 1000 == 0:
                self.save_model(epoch_num, batch_num)
        if epoch_num % self.save_every == 0:
            self.save_model(epoch_num + 1, 0)

    def training_step(self, epoch, batch_num, data):
        content, style = data
        content_loss, style_loss, stylized = self.model(data)
        loss = content_loss + self.k * style_loss

        step = epoch * len(self.dataloader) + batch_num
        self.logs(step, content, style, stylized, content_loss, style_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return content_loss, style_loss
    
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
            vutils.save_image([content[i].squeeze(), style[i].squeeze(), stylized[i].squeeze()], str(current) + ext, nrow = 1) 

    def save_model(self, epoch, step):
        checkpoint = {
            "model" : self.model.state_dict(),
            "epoch" : epoch,
            "optimizer": self.optimizer.state_dict(),
            "step": step
        }
        PATH = "checkpoint.pt"
        torch.save(checkpoint, PATH)

    def load_checkpoint(self):
        PATH = "checkpoint.pt"
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        print("RESUMING FROM EPOCH ", epoch)
        return epoch
    
    def evaluate(self, num):
        self.model.eval()
        progress_bar = trange(0, num, desc = "Evaluating model on images")
        for batch_num, data in enumerate(self.dataloader):
            if progress_bar.n >= num:
                break
            data = data.to(self.device)
            self.evaluation_step(data, progress_bar.n)
            progress_bar.update(self.dataloader.batch_size)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Style Transfer following the AdaIN paper by Huang, et. al.')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('load_model', type=int, help='0 to start a new model, non 0 to continue from checkpoint.pt')
    parser.add_argument('evaluate', type=int, help='0 to train, X to evaluate for X random image pairs')
    args = parser.parse_args() 
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Model(device)
    dataset = ContentStyleDataset('album_covers_512/', 'images/', transform = loader)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, num_workers = 8, collate_fn = collate)
    writer = SummaryWriter()
    trainer = Trainer(model, dataloader, learning_rate = 1e-4, save_every = 1, writer = writer)
    if args.load_model:
        trainer.load_checkpoint()
    if args.evaluate:
        trainer.evaluate(args.evaluate)
    else:
        trainer.train(args.total_epochs)
