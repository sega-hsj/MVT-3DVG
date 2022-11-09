import torch
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


print(torch.cuda.default_stream())