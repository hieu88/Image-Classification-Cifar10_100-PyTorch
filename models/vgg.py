import torch
import torch.nn as nn


config ={
    'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

class VGG(nn.Module):
    def __init__(self,in_c=3,num_classes=10,inp_height=32,inp_width=32,cfg = config['vgg11']):
        super(VGG,self).__init__()
        self.in_c = in_c
        self.feature_layers = self.make_feature_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512*(inp_height//32)*(inp_width//32),4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes)
        )

    def forward(self,x):
        x = self.feature_layers(x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        return x

    def make_feature_layers(self,config):
        in_c = self.in_c
        feature = []
        for cf in config:
            if cf == 'M':
                feature += [nn.MaxPool2d(kernel_size=2,stride=2)]
            
            else:
                feature += [nn.Conv2d(in_c,cf,kernel_size=3,stride=1,padding=1)]
                feature += [nn.BatchNorm2d(cf)]
                feature += [nn.ReLU(inplace=True)]
                in_c = cf
        return nn.Sequential(*feature)

def VGG11(inp_height,inp_width):
    return VGG(inp_height = inp_height,inp_width = inp_width,cfg = config['vgg11'])

def VGG13(inp_height,inp_width):
    return VGG(inp_height = inp_height,inp_width = inp_width,cfg = config['vgg13'])

def VGG16(inp_height,inp_width):
    return VGG(inp_height = inp_height,inp_width = inp_width,cfg = config['vgg16'])

def VGG19(inp_height,inp_width):
    return VGG(inp_height = inp_height,inp_width = inp_width,cfg = config['vgg19'])

# if __name__ == '__main__':
#     model = VGG11(64,64)
#     x = torch.randn(1,3,64,64)
#     print(model(x).shape)