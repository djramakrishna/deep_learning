import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter #allows to send data to tensorboard files 
from IPython.display import display, clear_output

from collections import OrderedDict
from collections import namedtuple
from itertools import product

torch.set_printoptions(linewidth = 120) 
torch.set_grad_enabled(True) 

train_set = torchvision.datasets.FashionMNIST(
    root='./data'  
    ,train=True    
    ,download=True 
    ,transform=transforms.Compose([
        transforms.ToTensor()
        #normalize
    ])
)

loader = DataLoader(train_set, batch_size = len(train_set), num_workers = 1)

train_set_normal = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, 
                                                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)]))

loader = DataLoader(train_set, batch_size = len(train_set), num_workers = 1)

class RunBuilder():
    @staticmethod #The main thing to note about using this class is that it has a static method called get_runs(). This method will get the runs for us that it builds based on the parameters we pass in.
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
        
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class Network(nn.Module):
        
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)


    def forward(self, t):
        
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride =2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride =2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
       
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t) 

        # (6) ouput layer
        t = self.out(t)
        return t



class RunManager():

    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None 
        self.run_count = 0
        self.run_data = []  #track param,results for each epoch for each run
        self.run_start_time = None  #run durations

        self.network = None #save network
        self.loader = None #save data loader for run
        self.tb = None #summarywriter for tensorboard


#Anytime we see this, we need to be thinking about removing these prefixes. (epoch_count, epoch_loss, ....)
# Data that belongs together should be together. 
# This is done by encapsulating the data inside of a class.
#this is done in next cell

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images.to(getattr(run, 'device', 'cpu')))     ####### getattr is modified - check which device the run for 

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        clear_output(wait=True) #specific to jy notebooks clear curr o/p and display new data frame
        display(df)
    
    def track_loss(self, loss):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
    
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @torch.no_grad()

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, fileName):

        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{fileName}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:json.dump(self.run_data, f, ensure_ascii=False, indent=4)


trainsets = {
    'not_normal' : train_set,
    'normal' : train_set_normal
}


params = OrderedDict(
    lr = [.01]
    ,batch_size = [1000, 10000, 20000]
    ,num_workers = [0,1]
    ,device = ['cuda', 'cpu']   
    ,trainset = ['not_normal', 'normal']
)

m = RunManager()

for run in RunBuilder.get_runs(params):
    
    device = torch.device(run.device) ## create a pytroch device and pass cpu or cuda
    network = Network().to(device) #initialize network on cuda
    train_loader = DataLoader(train_sets[run.train_set], batch_size = run.batch_size, num_workers = run.num_workers)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)

    m.begin_run(run, network, train_loader)

    for epoch in range(1):
        m.begin_epoch()
        for batch in train_loader:

                images = batch[0].to(device) #unpack them & send to device 
                labels = batch[1].to(device) #unpack them & send to device
                preds = network(images) 
                loss = F.cross_entropy(preds, labels) 
                optimizer.zero_grad()               
                loss.backward()          
                optimizer.step()
                
                m.track_loss(loss)
                m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()

m.save('results')

pd.DataFrame.from_dict(m.run_data).sort_values('accuracy', ascending = False)



