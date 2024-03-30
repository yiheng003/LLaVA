import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import linalg as LA
import math 

def patch_wise_similarity(tensor):
    tensor_n = LA.norm(tensor, dim=(0,2))[:, None]
    tensor_norm = torch.matmul(tensor_n, tensor_n.transpose(0,1))
    tensor_product = torch.matmul(tensor, tensor.transpose(1,2))
    tensor_sim = torch.div(tensor_product, tensor_norm)
    print(tensor_sim)
    tensor_sim_sum = torch.sum(tensor_sim)
    
    return tensor_sim_sum

# vector1 = torch.rand([1, 5, 25], dtype=torch.float)
vector1 = torch.tensor([[[1,2],[3,4]]], dtype=torch.float)
print("vector1: ",vector1)
out = patch_wise_similarity(vector1)
print(out)

# proj_vector = torch.rand([1, 25, 5], dtype=torch.float)
proj_vector = torch.tensor([[[9,3],[4,5]]], dtype=torch.float)
vector2 = torch.matmul(vector1, proj_vector)
print("vector2: ",vector2)
out = patch_wise_similarity(vector2)
print(out)

# A = torch.rand((1,576, 576),dtype=torch.float)
# print(torch.sum(A))