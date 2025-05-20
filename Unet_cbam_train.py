import os

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from logconfig import Logger
from config import Config as conf
from models.point_unt import UNet
from early_stopping import EarlyStopping
import matplotlib.pyplot as plt

from models.unet_cbam import U_Net_v1

EPOCH = 'epoch'
train_size = 'train_size'
batch_size = 2
val_size = 'val_size'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
loss_f = nn.BCELoss().to(device)

trainloss_log = Logger('log path', level='debug')
valloss_log = Logger('log path', level='debug')

#2018
img = np.load('your path')
label = np.load('your path')

print(img.shape,label.shape)

state = np.random.get_state()
np.random.shuffle(img)
np.random.set_state(state)
np.random.shuffle(label)

#1000,1,128,128
img_tensor = torch.FloatTensor(img).unsqueeze(1).to(device)
#1000,3,128,128
label_tensor = torch.FloatTensor(label).to(device)
print(img_tensor.shape)
print(label_tensor.shape)

train_img = img_tensor[0:train_size]
train_label = label_tensor[0:train_size]

val_img = img_tensor[train_size:val_size]
val_label = label_tensor[train_size:val_size]

net = U_Net_v1().to(device)


# optimizer = optim.Adam(net.parameters(),lr=0.01)
# optimizer = torch.optim.Adam(net.parameters(), lr=conf.lr, weight_decay=conf.decay)
# scheduler = lr_scheduler.StepLR(optimizer,
#                                     step_size=conf.step_size,
#                                     gamma=conf.gamma)

optimizer=optim.Adam(net.parameters())
scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.1)

def train():
    for epoch in range(EPOCH):
        net.train()
        sum_loss = 0
        s = 0
        epoch = epoch + 1
        for b in range(0,train_size,batch_size):
            start_idx = b
            end_idx = (b + batch_size) if (b + batch_size) < train_size else train_size
            input_img = train_img[start_idx:end_idx]

            input_label = train_label[start_idx:end_idx]

            out = net(input_img)

            loss = loss_f(out, input_label)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss
            s = s + 1
        e_loss = sum_loss / s
        print("epoch: {}, loss: {}".format(epoch, e_loss))
        trainloss_log.logger.info('epoch: {},loss: {}'.format(epoch, e_loss))
        if epoch%5==0 and epoch>19:
            torch.save(net.state_dict(),"save_path"+str(epoch)+".pth")

        net.eval()
        val_sum_loss = 0
        val_s = 0

        for b1 in range(0, val_size, batch_size):
            start_val_idx = b1
            end_val_idx = (b1 + batch_size) if (b1 + batch_size) < val_size else val_size
            input_val_img = img_tensor[start_val_idx:end_val_idx]
            # poi_val = poi_tensor[start_val_idx:end_val_idx]
            input_val_label = label_tensor[start_val_idx:end_val_idx]

            with torch.no_grad():
                val_out = net(input_val_img)
                val_loss = loss_f(val_out, input_val_label)
                val_sum_loss += val_loss
                val_s = val_s + 1
        eval_loss = val_sum_loss / val_s
        print("epoch: {}, val_loss: {}".format(epoch, eval_loss))
        valloss_log.logger.info('epoch: {},val_loss: {}'.format(epoch, eval_loss ))


        early_stopping(eval_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("finish")

if __name__ == '__main__':
    save_path = "your path"
    early_stopping = EarlyStopping(save_path)
    print(torch.cuda.is_available())

    train()
