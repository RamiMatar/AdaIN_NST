import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import tqdm.notebook as tq
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from dataset import collate, ContentStyleDataset, loader
from net import Model
class Trainer():
    def __init__(self, model, dataloader, learning_rate, save_every, writer, optimizer = "adam", decay_lr = None, alpha = 1.0, k = 10.0):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.cuda()
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
        with tq.trange(epoch, num_epochs + 1, desc="All epochs") as epochs:
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
        progress_bar = tq.trange(num_batches, desc = "epoch" + str(epoch_num))
        for batch_num, data in enumerate(self.dataloader):
            data = data.to(self.device)
            content_loss, style_loss = self.training_step(epoch_num, batch_num, data)
            progress_bar.update(1)
            progress_bar.set_postfix(content_loss = content_loss.detach().cpu().numpy(), style_loss = style_loss.detach().cpu().numpy(), total_loss = (content_loss + style_loss).detach().cpu().numpy())
        if epoch_num % self.save_every == 0:
            self.save_model(epoch_num)

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

    def save_model(self, epoch):
        checkpoint = {
            "model" : self.model.state_dict(),
            "epoch" : epoch + 1,
            "optimizer": self.optimizer
        }
        PATH = "checkpoint.pt"
        torch.save(checkpoint, PATH)

    def load_checkpoint(self):
        PATH = "checkpoint.pt"
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print("RESUMING FROM EPOCH ", epoch)
        return epoch
    
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Model(device)
    dataset = ContentStyleDataset('album_covers_512/', 'images/', transform = loader)
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 4, collate_fn = collate)
    writer = SummaryWriter()
    trainer = Trainer(model, dataloader, learning_rate = 1e-3, save_every = 1, writer = writer)
    trainer.train(10)