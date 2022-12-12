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

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0, ext_attn_list=None, ext_v_list=None, ext_y_list=None):
        if method == 'igrad':
            grads_all = 0
            for alpha in np.linspace(0, 1, 20):
                output = self.model(input * alpha)
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)

                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0, index] = 1
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * output)

                self.model.zero_grad()
                one_hot.backward(retain_graph=True)
                grads = []
                for blk in self.model.blocks:
                    grad = blk.attn.get_attn_gradients()
                    grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                    grads.append(grad)
                grads_all += torch.stack(grads)
            kwargs = {"alpha": 1}
            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method='igrad', is_ablation=is_ablation,
                                  start_layer=start_layer, grads_all=grads_all, **kwargs)
        elif method == 'perturb':
            outputs = []
            output = self.model(input)
            ext_attn_list = None
            ext_v_list = None
            base = output.max()
            if index == None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)
            for i in range(1, 197):
                with torch.no_grad():
                    output_ = self.model(input, keeptoken=i, ext_attn_list=ext_attn_list, ext_v_list=ext_v_list, start_layer=11)
#                 outputs.append(base - output_[0, index])
                outputs.append(output_[0, index])
            ret = torch.stack(outputs)
            return ret / ret.max()

        elif method == 'perturb_all':
            outputs = []
            with torch.no_grad():
                for i in range(1, 197):
                    output_ = self.model(input, keeptoken=i, ext_attn_list=ext_attn_list, ext_v_list=ext_v_list, start_layer=start_layer, ext_y_list=ext_y_list, ext_x1_list=None, ext_grad_list=None)
                    outputs.append(output_[0, index])
            ret = torch.stack(outputs)
            return ret / ret.max()
        
        elif method == 'perturb_all_fast':
            with torch.no_grad():
                output = self.model.fast_forward(input, ext_attn_list=ext_attn_list, ext_v_list=ext_v_list, start_layer=start_layer, ext_y_list=ext_y_list, ext_x1_list=None, ext_grad_list=None)
            ret = output[:, index]
            return ret / ret.max()
        
        elif method == 'perturb_patch':
            outputs = []
            for i in range(1, 197):
                with torch.no_grad():
                    output_ = self.model(input, keep_patch=i)
                outputs.append(output_[0, index])
            ret = torch.stack(outputs)
            return ret / ret.max()
        
        elif method == 'perturb_patch_all':
            outputs = []
            for i in range(1, 197):
                with torch.no_grad():
                    output_ = self.model(input, keeptoken=i, ext_attn_list=ext_attn_list, ext_v_list=ext_v_list, start_layer=start_layer, ext_y_list=ext_y_list, ext_x1_list=None, ext_grad_list=None, keep_patch=i)
                outputs.append(output_[0, index])
            ret = torch.stack(outputs)
            return ret / ret.max()

        elif method == 'perturb_zero':
            outputs = []
#             with torch.no_grad():
            output = self.model(input)

            base = output.max()
            if index == None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)

            self.model.zero_grad()
            bg_output = self.model(input*0.)
            one_hot = np.zeros((1, bg_output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * bg_output)
            one_hot.backward(retain_graph=True)
            base_attn=self.model.blocks[-1].attn.get_attn()
            for i in range(1, 197):
                with torch.no_grad():
#                     pdb.set_trace()
                    output_ = self.model(input, keeptoken=i, base_attn=base_attn)
#                     output_ = self.model(input, keeptoken=i)
                outputs.append(output_[0, index])
            ret = torch.stack(outputs)
#             pdb.set_trace()
            return ret / ret.max()

        elif method == 'taylor':
            with torch.no_grad():
                output = self.model(input)
                kwargs = {"alpha": 1}
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)

            self.model.zero_grad()
            output = self.model(input * 1.)
#             pdb.set_trace()
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            grads=None
            
#             self.model.zero_grad()
#             grads = nth_derivative(one_hot, self.model, 1)
#             torch.cuda.empty_cache()
            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation, start_layer=start_layer, nth_grads=grads, x0=None, **kwargs)
        
        elif method == 'taylor_2nd_order':
            with torch.no_grad():
                output = self.model(input)
                kwargs = {"alpha": 1}
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)
            grads = []
            attns = []
            for scalor in [1.]:
                self.model.zero_grad()
                output = self.model(input * scalor * 1)
                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0, index] = 1
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * output)
                attns.append(self.model.blocks[-1].attn.get_attn()[0].clamp(0).mean(0)[0, 1:])
                self.model.zero_grad()
#                 grads.append(nth_derivative(one_hot, self.model, 1)[0][0].mean(0)[0, 1:])
                grads.append(nth_derivative(one_hot, self.model, 1)[0][0][:, 0, 1:])
#             grad_2 = torch.stack([torch.autograd.grad(grads[0][i], self.model.blocks[-1].attn.get_attn(), retain_graph=True)[0][0].sum(0)[0, 1:][i] for i in range(len(grads[0]))])
            grads_2 = torch.zeros_like(grads[0])
            for h in range(grads[0].shape[0]):
                for t in range(grads[0].shape[1]):
                    
                    grads_2[h, t] = torch.autograd.grad(grads[0][h, t], self.model.blocks[-1].attn.get_attn(), retain_graph=True)[0][0, :, 0, 1:][h, t]
#             pdb.set_trace()
            torch.cuda.empty_cache()
            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation, start_layer=start_layer, nth_grads=[grads[0], grads_2], x0=None, **kwargs)
    
        elif method == 'taylor_2nd_order_0.5':
            with torch.no_grad():
                output = self.model(input)
                kwargs = {"alpha": 1}
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)
            grads = []
            attns = []
            for scalor in [1.]:
                self.model.zero_grad()
                output = self.model(input * scalor * 0.5)
                one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
                one_hot[0, index] = 1
                one_hot_vector = one_hot
                one_hot = torch.from_numpy(one_hot).requires_grad_(True)
                one_hot = torch.sum(one_hot.cuda() * output)
                attns.append(self.model.blocks[-1].attn.get_attn()[0].clamp(0).mean(0)[0, 1:])
                self.model.zero_grad()
#                 grads.append(nth_derivative(one_hot, self.model, 1)[0][0].mean(0)[0, 1:])
                grads.append(nth_derivative(one_hot, self.model, 1)[0][0][:, 0, 1:])
#             grad_2 = torch.stack([torch.autograd.grad(grads[0][i], self.model.blocks[-1].attn.get_attn(), retain_graph=True)[0][0].sum(0)[0, 1:][i] for i in range(len(grads[0]))])
            grads_2 = torch.zeros_like(grads[0])
            for h in range(grads[0].shape[0]):
                for t in range(grads[0].shape[1]):
                    
                    grads_2[h, t] = torch.autograd.grad(grads[0][h, t], self.model.blocks[-1].attn.get_attn(), retain_graph=True)[0][0, :, 0, 1:][h, t]
#             pdb.set_trace()
            torch.cuda.empty_cache()
            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method='taylor_2nd_order', is_ablation=is_ablation, start_layer=start_layer, nth_grads=[grads[0], grads_2], x0=None, **kwargs)

        elif method == 'taylor_cam':
            with torch.no_grad():
                output = self.model(input)
                kwargs = {"alpha": 1}
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)

            self.model.zero_grad()
            output = self.model(input * 0.1)
#             pdb.set_trace()
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            
#             one_hot.backward(create_graph=True)
#             pdb.set_trace()
#             x0 = self.model.blocks[-1].attn.get_attn()[0]
            x0 = 0
            grads = nth_derivative(one_hot, self.model, 1)
            torch.cuda.empty_cache()            
#             one_hot.backward()
            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method='taylor_cam', is_ablation=is_ablation, start_layer=start_layer, nth_grads=grads, x0=x0, **kwargs)


        elif method == 'taylor_cam_0.5':
            with torch.no_grad():
                output = self.model(input)
                kwargs = {"alpha": 1}
                if index == None:
                    index = np.argmax(output.cpu().data.numpy(), axis=-1)

            self.model.zero_grad()
            output = self.model(input * 0.5)
#             pdb.set_trace()
            
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            
#             one_hot.backward(create_graph=True)
#             pdb.set_trace()
#             x0 = self.model.blocks[-1].attn.get_attn()[0]
            x0 = 0
            grads = nth_derivative(one_hot, self.model, 1)
            torch.cuda.empty_cache()            
#             one_hot.backward()
            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method='taylor_cam', is_ablation=is_ablation, start_layer=start_layer, nth_grads=grads, x0=x0, **kwargs)

        else:
            output = self.model(input)
            kwargs = {"alpha": 1}
            if index == None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation, start_layer=start_layer, **kwargs)
            



class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None, scalor=1):
        output = self.model(input.cuda()*scalor, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]
