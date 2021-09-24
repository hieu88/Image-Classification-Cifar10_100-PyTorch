import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config

def save_checkpoints(state,model_folder,model_name):
    torch.save(state,os.path.join(model_folder,model_name))

def loss_acc_hist(hist_path):
    hist = torch.load(hist_path,map_location="cpu")
    train_losses = hist["train_losses"]
    test_losses = hist["test_losses"]
    train_accuracies = hist["train_accuracies"]
    test_accuracies = hist["test_accuracies"]
    return train_losses,test_losses,train_accuracies,test_accuracies

def visualize_loss(train_losses,test_losses,training_epochs):
    num_epoch_steps = len(train_losses)
    epoch_range = np.arange(0,training_epochs+Config.EVAL_TRAIN_STEP.value,Config.EVAL_TRAIN_STEP.value)
    train_loss_hist = []
    test_loss_hist = []
    for key in train_losses.keys():
        train_loss_hist.append(train_losses[key])
        test_loss_hist.append(test_losses[key])
    plt.plot(epoch_range,np.array(train_loss_hist,dtype=np.float32))
    plt.plot(epoch_range,np.array(test_loss_hist,dtype=np.float32))

def visualize_acc(train_accuracies,test_accuracies,training_epochs):
    num_epoch_steps = len(train_accuracies)
    epoch_range = np.arange(0,training_epochs+Config.EVAL_TRAIN_STEP.value,Config.EVAL_TRAIN_STEP.value)
    train_acc_hist = []
    test_acc_hist = []
    for key in train_accuracies.keys():
        train_acc_hist.append(train_accuracies[key])
        test_acc_hist.append(test_accuracies[key])
    plt.plot(epoch_range,np.array(train_acc_hist,dtype=np.float32))
    plt.plot(epoch_range,np.array(test_acc_hist,dtype=np.float32))