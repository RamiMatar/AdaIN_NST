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
from dataset import collate, ContentStyleDataset, loader, denorm
from net import Model
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler



class Trainer():
    def __init__(self, rank, model, dataloader, learning_rate, save_every, writer, progress_dir, checkpoint_path, checkpoint_save_path, optimizer = "adam", lr_decay = 5e-5, alpha = 1.0, k = 5.0):
        self.device = torch.device(rank)
        self.rank = rank
        self.checkpoint_path = checkpoint_path
        self.checkpoint_save_path = checkpoint_save_path
        self.model = DDP(model, device_ids=[rank])
        self.dataloader = dataloader
        self.lr = learning_rate
        self.save_every = save_every
        self.writer = writer
        self.progress_dir = progress_dir + '/'
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.module.decoder.parameters(), lr = self.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.module.decoder.parameters(), lr = self.lr)
        self.lr_decay = lr_decay
        self.alpha = alpha
        self.k = k
        self.epoch = 0
        self.step = 0

    def train(self, num_epochs):
        with trange(self.epoch, num_epochs, desc="All epochs") as epochs:
            for epoch in epochs:
                self.run_epoch(epoch)

    def logs(self, step, content, style, stylized, content_loss, style_loss, loss):
        self.writer.add_scalar("Style Loss", style_loss, step)
        self.writer.add_scalar("Content Loss", content_loss, step)
        self.writer.add_scalar("Total Loss", loss, step)
        if step % 10 == 0:
            grid = vutils.make_grid([denorm(content[0].squeeze(), self.device), denorm(style[0].squeeze(), self.device), denorm(stylized[0].squeeze(), self.device)], nrow = 3) 
            self.writer.add_image("Style Transfer Result", grid, step)
            vutils.save_image(grid, self.progress_dir + str(self.step)+'.png')

    def learning_rate_decay(self):
        '''Learning rate decay based on the torch implementation by the authors of the paper '''
        lr = self.lr / (1.0 + self.lr_decay * self.step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def overfit(self, iters):
        '''Overfit the model on a single batch for debugging purposes, sanity check that the encoder is set up properly and backprop is doing its thing'''
        data = next(iter(self.dataloader))
        data = data.to(self.device)
        with trange(0, iters, desc="All epochs") as progress_bar:
            for i in range(iters):
                self.training_step(data)
                content_loss, style_loss, loss = self.training_step(data)
                progress_bar.update(1)

             
    def run_epoch(self, epoch_num):
        num_batches = len(self.dataloader)
        progress_bar = trange(num_batches, desc = "epoch" + str(epoch_num) + " gpu: " + str(self.device))
        for batch_num, data in enumerate(self.dataloader):
            self.learning_rate_decay()
            data = data.to(self.device)
            content_loss, style_loss, loss = self.training_step(data)
            progress_bar.update(1)
            progress_bar.set_postfix(content_loss = content_loss.item(), style_loss = style_loss.item(), total_loss = loss.item())
            if batch_num % 200 == 0 and self.rank == 0:
                self.save_model(epoch_num)

    def training_step(self, data):
        content, style = data
        content_loss, style_loss, stylized = self.model(data)
        loss = content_loss + self.k * style_loss
        
        self.step += 1
        if self.rank == 0:
            self.logs(self.step, content, style, stylized, content_loss, style_loss, loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return content_loss, style_loss, loss
    
    def save_model(self, epoch):
        checkpoint = {
            "model" : self.model.state_dict(),
            "epoch" : epoch + 1,
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        torch.save(checkpoint, self.checkpoint_save_path)

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location = "cuda:{}".format(self.rank))
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
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    ddp_setup(rank, world_size)
    model, dataloader, writer = load_train_objects(rank, args.writer_dir)
    trainer = Trainer(rank, model, dataloader, learning_rate = 1e-4, save_every = 1, writer = writer, progress_dir = args.progress_dir, checkpoint_path = args.checkpoint_path, checkpoint_save_path = args.checkpoint_save_path)
    if args.load_model:
        trainer.load_checkpoint()
    if args.overfit == 'overfit':
        trainer.overfit(10000)
    else:
        trainer.train(args.total_epochs)
    destroy_process_group()

def load_train_objects(rank, writer_dir):
    model = Model(rank)
    dataset = ContentStyleDataset('content/', 'style/', transform = loader)
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = collate, shuffle = False, num_workers = 0, sampler = DistributedSampler(dataset))
    writer = SummaryWriter(writer_dir)
    return model, dataloader, writer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Style Transfer following the AdaIN paper by Huang, et. al.')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model', default = 1)
    parser.add_argument('--load_model', type=int, help='0 to start a new model, non 0 to continue from checkpoint.pt, default 0', default = 0)
    parser.add_argument('--writer_dir', type=str, help='Directory to save tensorboard logs, default is ./runs', default = './runs')
    parser.add_argument('--overfit', type=str, help="overfit to overfit, any other str to run normally", default = "regular")
    parser.add_argument('--progress_dir', type=str, help="Directory to save progress images, default is ./outputs", default = "./outputs")
    parser.add_argument('--save', type = bool, help = 'True by default, meaning the model saves checkpoints while training.', default = True)
    parser.add_argument('--checkpoint_path', type = str, help = 'path for the checkpoint to load model from, default is \'checkpoint.pt\'', default = 'checkpoint.pt')
    parser.add_argument('--checkpoint_save_path', type = str, help = 'path to save model checkpoints in this run, default is \'checkpoint.pt\'', default = 'checkpoint.pt')
    args = parser.parse_args() 
    
    world_size = torch.cuda.device_count()
    print("Starting multi processes")
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
