import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import DCN
from models.SLFNet.modulated_deform_conv_func import ModulatedDeformConvFunction

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("DCN import ok")
print("SLFNet DCN wrapper import ok", ModulatedDeformConvFunction.__name__)
