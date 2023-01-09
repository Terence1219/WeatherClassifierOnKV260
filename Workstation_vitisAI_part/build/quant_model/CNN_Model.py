# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #CNN_Model::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[4, 4], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[conv1]/Conv2d[0]/22
        self.module_2 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[conv1]/MaxPool2d[1]/input.2
        self.module_3 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[4, 4], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[conv2]/Conv2d[0]/38
        self.module_4 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[conv2]/MaxPool2d[1]/input.3
        self.module_5 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[4, 4], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[conv3]/Conv2d[0]/54
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[conv3]/MaxPool2d[1]/input.4
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[4, 4], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #CNN_Model::CNN_Model/Sequential[conv4]/Conv2d[0]/70
        self.module_8 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #CNN_Model::CNN_Model/Sequential[conv4]/MaxPool2d[1]/76
        self.module_9 = py_nndct.nn.Module('shape') #CNN_Model::CNN_Model/78
        self.module_10 = py_nndct.nn.Module('reshape') #CNN_Model::CNN_Model/input.5
        self.module_11 = py_nndct.nn.Linear(in_features=31968, out_features=200, bias=True) #CNN_Model::CNN_Model/Linear[fc1]/input
        self.module_12 = py_nndct.nn.Linear(in_features=200, out_features=18, bias=True) #CNN_Model::CNN_Model/Linear[fc2]/89

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_5)
        self.output_module_7 = self.module_7(self.output_module_6)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_9 = self.module_9(input=self.output_module_8, dim=0)
        self.output_module_10 = self.module_10(input=self.output_module_8, size=[self.output_module_9,-1])
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        return self.output_module_12
