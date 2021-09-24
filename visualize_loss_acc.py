import matplotlib.pyplot as plt
from utils import loss_acc_hist,visualize_loss,visualize_acc

if __name__ == '__main__':
    vgg13_loss_acc_hist = loss_acc_hist("D:\Image-Classification-Cifar10_100-PyTorch\model_folder\\vgg\\vgg13_loss_acc.pth.tar")
    visualize_loss(vgg13_loss_acc_hist[0],vgg13_loss_acc_hist[1],300)
    plt.show()
    visualize_acc(vgg13_loss_acc_hist[2],vgg13_loss_acc_hist[3],300)
    plt.show()