# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Net::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[7, 7], stride=[2, 2], padding=[3, 3], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Conv2d[conv1]/input.2
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/ReLU[relu]/139
        self.module_4 = py_nndct.nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False) #Net::Net/ResNet[resnet18]/MaxPool2d[maxpool]/input.4
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[0]/Conv2d[conv1]/input.5
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/input.7
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[0]/Conv2d[conv2]/input.8
        self.module_10 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[0]/input.9
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/input.10
        self.module_12 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[1]/Conv2d[conv1]/input.11
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/input.13
        self.module_15 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[1]/Conv2d[conv2]/input.14
        self.module_17 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[1]/input.15
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/input.16
        self.module_19 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[0]/Conv2d[conv1]/input.17
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/input.19
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[0]/Conv2d[conv2]/input.20
        self.module_24 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.21
        self.module_26 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[0]/input.22
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/input.23
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[1]/Conv2d[conv1]/input.24
        self.module_30 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/input.26
        self.module_31 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[1]/Conv2d[conv2]/input.27
        self.module_33 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[1]/input.28
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/input.29
        self.module_35 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[0]/Conv2d[conv1]/input.30
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/input.32
        self.module_38 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[0]/Conv2d[conv2]/input.33
        self.module_40 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.34
        self.module_42 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[0]/input.35
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/input.36
        self.module_44 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[1]/Conv2d[conv1]/input.37
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/input.39
        self.module_47 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[1]/Conv2d[conv2]/input.40
        self.module_49 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[1]/input.41
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/input.42
        self.module_51 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[0]/Conv2d[conv1]/input.43
        self.module_53 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/input.45
        self.module_54 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[0]/Conv2d[conv2]/input.46
        self.module_56 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/Conv2d[0]/input.47
        self.module_58 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[0]/input.48
        self.module_59 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/input.49
        self.module_60 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[1]/Conv2d[conv1]/input.50
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/input.52
        self.module_63 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[1]/Conv2d[conv2]/input.53
        self.module_65 = py_nndct.nn.Add() #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[1]/input.54
        self.module_66 = py_nndct.nn.ReLU(inplace=True) #Net::Net/ResNet[resnet18]/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/input.55
        self.module_67 = py_nndct.nn.AdaptiveAvgPool2d(output_size=1) #Net::Net/ResNet[resnet18]/AdaptiveAvgPool2d[avgpool]/483
        self.module_68 = py_nndct.nn.Module('flatten') #Net::Net/ResNet[resnet18]/input
        self.module_69 = py_nndct.nn.Linear(in_features=512, out_features=3, bias=True) #Net::Net/ResNet[resnet18]/Linear[fc]/490

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_3 = self.module_3(self.output_module_1)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        self.output_module_7 = self.module_7(self.output_module_5)
        self.output_module_8 = self.module_8(self.output_module_7)
        self.output_module_10 = self.module_10(input=self.output_module_8, alpha=1, other=self.output_module_4)
        self.output_module_11 = self.module_11(self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_11)
        self.output_module_14 = self.module_14(self.output_module_12)
        self.output_module_15 = self.module_15(self.output_module_14)
        self.output_module_17 = self.module_17(input=self.output_module_15, alpha=1, other=self.output_module_11)
        self.output_module_18 = self.module_18(self.output_module_17)
        self.output_module_19 = self.module_19(self.output_module_18)
        self.output_module_21 = self.module_21(self.output_module_19)
        self.output_module_22 = self.module_22(self.output_module_21)
        self.output_module_24 = self.module_24(self.output_module_18)
        self.output_module_26 = self.module_26(input=self.output_module_22, alpha=1, other=self.output_module_24)
        self.output_module_27 = self.module_27(self.output_module_26)
        self.output_module_28 = self.module_28(self.output_module_27)
        self.output_module_30 = self.module_30(self.output_module_28)
        self.output_module_31 = self.module_31(self.output_module_30)
        self.output_module_33 = self.module_33(input=self.output_module_31, alpha=1, other=self.output_module_27)
        self.output_module_34 = self.module_34(self.output_module_33)
        self.output_module_35 = self.module_35(self.output_module_34)
        self.output_module_37 = self.module_37(self.output_module_35)
        self.output_module_38 = self.module_38(self.output_module_37)
        self.output_module_40 = self.module_40(self.output_module_34)
        self.output_module_42 = self.module_42(input=self.output_module_38, alpha=1, other=self.output_module_40)
        self.output_module_43 = self.module_43(self.output_module_42)
        self.output_module_44 = self.module_44(self.output_module_43)
        self.output_module_46 = self.module_46(self.output_module_44)
        self.output_module_47 = self.module_47(self.output_module_46)
        self.output_module_49 = self.module_49(input=self.output_module_47, alpha=1, other=self.output_module_43)
        self.output_module_50 = self.module_50(self.output_module_49)
        self.output_module_51 = self.module_51(self.output_module_50)
        self.output_module_53 = self.module_53(self.output_module_51)
        self.output_module_54 = self.module_54(self.output_module_53)
        self.output_module_56 = self.module_56(self.output_module_50)
        self.output_module_58 = self.module_58(input=self.output_module_54, alpha=1, other=self.output_module_56)
        self.output_module_59 = self.module_59(self.output_module_58)
        self.output_module_60 = self.module_60(self.output_module_59)
        self.output_module_62 = self.module_62(self.output_module_60)
        self.output_module_63 = self.module_63(self.output_module_62)
        self.output_module_65 = self.module_65(input=self.output_module_63, alpha=1, other=self.output_module_59)
        self.output_module_66 = self.module_66(self.output_module_65)
        self.output_module_67 = self.module_67(self.output_module_66)
        self.output_module_68 = self.module_68(input=self.output_module_67, start_dim=1, end_dim=3)
        self.output_module_69 = self.module_69(self.output_module_68)
        return self.output_module_69
