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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

class Trainer():
    def __init__(self, rank, model, dataloader, learning_rate, save_every, writer, optimizer = "adam", decay_lr = None, alpha = 1.0, k = 1.0):
        self.device = torch.device(rank)
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
        self.epoch = 0
        self.step = 0

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
        grid = vutils.make_grid([content[0].squeeze(), style[0].squeeze(), stylized[0].squeeze()], nrow = 1) 
        self.writer.add_image("Style Transfer Result", grid, step)
        
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

        self.step += 1
        self.logs(self.step, content, style, stylized, content_loss, style_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return content_loss, style_loss
    
    def save_model(self, epoch, step):
        checkpoint = {
            "model" : self.model.state_dict(),
            "epoch" : self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        PATH = "checkpoint.pt"
        torch.save(checkpoint, PATH)

    def load_checkpoint(self):
        PATH = "checkpoint.pt"
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        print("RESUMING FROM EPOCH ", self.epoch)
    

    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12974"
    init_process_group(backend='gloo', rank=rank, world_size=world_size)

def main(rank, world_size, args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    ddp_setup(rank, world_size)
    model, dataset, dataloader, writer = load_train_objects(rank)
    trainer = Trainer(model, dataloader, learning_rate = 1e-4, save_every = 1, writer = writer)
    if args.load_model:
        trainer.load_checkpoint()
    trainer.train(args.total_epochs)
    destroy_process_group()

def load_train_objects(rank):
    model = Model(rank)
    dataset = ContentStyleDataset('album_covers_512/', 'images/', transform = loader)
    dataloader = DataLoader(dataset, batch_size = 8, shuffle = True, num_workers = 4, sampler = DistributedSampler(dataset))
    writer = SummaryWriter()
    return model, dataset, dataloader, writer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Style Transfer following the AdaIN paper by Huang, et. al.')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('load_model', type=int, help='0 to start a new model, non 0 to continue from checkpoint.pt')
    args = parser.parse_args() 
    
    world_size = torch.cuda.device_count()
    print("Starting multi processes")
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
