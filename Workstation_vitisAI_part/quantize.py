'''
This script is based on the example:
https://github.com/Xilinx/Vitis-AI-Tutorials/blob/master/Design_Tutorials/09-mnist_pyt/files/quantize.py
'''

import os
import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

from common import *
from resnet18 import *

DIVIDER = '-----------------------------------------'


def quantize(build_dir, quant_mode, batchsize):

    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    # detect if a GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    # load trained model
    model = Net().to(device)
    model.load_state_dict(torch.load(os.path.join(float_model, 'weather_resnet18.pth'), map_location=device))

    # override batchsize if in test mode
    if (quant_mode=='test'):
        batchsize = 1

    rand_in = torch.randn([batchsize, 3, 224, 224])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model

    '''
    
    '''
    # dataset
    test_data = datasets.ImageFolder(
        'dataset/test_data_weather',
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])                         
    )

    # data loaders
    test_loader = DataLoader(dataset=test_data, 
                             batch_size=batchsize, 
                             shuffle=True,
                             num_workers=2)

    # evaluate quantized model
    test(quantized_model, device, test_loader)
    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    # convert to xmodel
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return


def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=16,         help='Testing batchsize - must be an integer. Default is 16')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantize(args.build_dir, args.quant_mode, args.batchsize)

  return


if __name__ == '__main__':
    run_main()

