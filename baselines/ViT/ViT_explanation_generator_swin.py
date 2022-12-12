import argparse
import torch
import numpy as np
from numpy import *
from torch.autograd.functional import hessian
import pdb

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

def nth_derivative(f, model, n):
    nth_grads = []
    for i in range(n):
        grads = torch.autograd.grad(f, model.blocks[-1].attn.get_attn(), create_graph=True)[0]
        nth_grads.append(grads)
        model.zero_grad()
        f = grads.mean()
    del f
    del model
    del grads
    torch.cuda.empty_cache()
    return nth_grads

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0, k=None, row=None):
        with torch.no_grad():
            output = self.model.fast_forward(input, k)
#             output = self.model.f(input, row)
        ret = output[:, index]
        return ret
        