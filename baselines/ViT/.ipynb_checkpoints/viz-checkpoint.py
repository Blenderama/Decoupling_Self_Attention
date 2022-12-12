import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
from tqdm import tqdm
from utils.metrices import *

from utils import render
from utils.saver import Saver
from utils.iou import IoU

from data.Imagenet import Imagenet_Segmentation

from ViT_explanation_generator import Baselines, LRP
from ViT_new import vit_base_patch16_224
from ViT_LRP import vit_base_patch16_224 as vit_LRP
from ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import pdb

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

plt.switch_backend('agg')

# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
#                     choices=[ 'rollout', 'lrp','transformer_attribution', 'transformer_attribution_last', 'transformer_attribution_both_last', 'transformer_attribution_2nd_last', 'transformer_attribution_cam_first', 'transformer_attribution_plus', 'transformer_attribution_dot', 'full_lrp', 'lrp_last_layer', 'attn_last_layer', 'attn_gradcam', 't_cam', 't_cam_diff', 't_cam_grad', 'igrad', 'igrad2', 'igrad3', 'transformer_attribution_markov', 'perturb', 'perturb_all', 'taylor_cam', 'taylor', 'taylor_2nd_order', 'taylor_cam_0.5', 'taylor_2nd_order_0.5', 'attn_gradcam_0.5', 'attn_gradcam_0.1'],
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
                    default=True,
                    help='')
parser.add_argument('--no-ia', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fgx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--is-ablation', type=bool,
                    default=False,
                    help='')
parser.add_argument('--imagenet-seg-path', type=str, required=True)
args = parser.parse_args()

args.checkname = args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define Saver
saver = Saver(args)
saver.results_dir = os.path.join(saver.experiment_dir, 'results')
if not os.path.exists(saver.results_dir):
    os.makedirs(saver.results_dir)
if not os.path.exists(os.path.join(saver.results_dir, 'input')):
    os.makedirs(os.path.join(saver.results_dir, 'input'))
if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
    os.makedirs(os.path.join(saver.results_dir, 'explain'))

args.exp_img_path = os.path.join(saver.results_dir, 'viz')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=test_img_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model
model = vit_base_patch16_224(pretrained=True).cuda()
baselines = Baselines(model)

if args.method in ['lrp_last_layer']:
    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)
# else:
    # LRP
model_LRP = vit_LRP(pretrained=True).cuda()
model_LRP.eval()
lrp = LRP(model_LRP)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()

    image.requires_grad = True

    image = image.requires_grad_()

    predictions = evaluator(image)
    
#     pdb.set_trace()
    output = lrp.model(image.cuda())
#     if index == None:
    cls_index = np.argmax(output.cpu().data.numpy(), axis=-1)
    lrp.model.zero_grad()
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, cls_index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    one_hot.backward(retain_graph=True)
    
    # segmentation test for the rollout baseline
    if args.method == 'rollout':
        Res = baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)
#     elif args.method == 'rollout':
#         Res = baselines.generate_rollout(image.cuda(), start_layer=1).reshape(batch_size, 1, 14, 14)

    # segmentation test for the LRP baseline (this is full LRP, not partial)
    elif args.method == 'full_lrp':
        Res = orig_lrp.generate_LRP(image.cuda(), method="full").reshape(batch_size, 1, 224, 224)
    
    # segmentation test for our method
    elif args.method == 'transformer_attribution':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution").reshape(batch_size, 1, 14, 14)
    elif args.method == 'sas':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="sas").reshape(batch_size, 1, 14, 14)

    elif args.method == 'transformer_attribution_0.5':
        Res = lrp.generate_LRP(image.cuda()*0.5, start_layer=1, method="transformer_attribution").reshape(batch_size, 1, 14, 14)
    elif args.method == 'transformer_attribution_0.1':
        Res = lrp.generate_LRP(image.cuda()*0.1, start_layer=1, method="transformer_attribution").reshape(batch_size, 1, 14, 14)

    elif args.method == 'transformer_attribution_last':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_last").reshape(batch_size, 1, 14, 14)
        
    elif args.method == 'transformer_attribution_2nd_last':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_2nd_last").reshape(batch_size, 1, 14, 14)

    elif args.method == 'transformer_attribution_both_last':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_both_last").reshape(batch_size, 1, 14, 14)
        
    elif args.method == 'transformer_attribution_cam_first':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_cam_first").reshape(batch_size, 1, 14, 14)

    elif args.method == 'transformer_attribution_plus':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_plus").reshape(batch_size, 1, 14, 14)

    elif args.method == 'transformer_attribution_markov':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_markov").reshape(batch_size, 1, 14, 14)

    elif args.method == 'transformer_attribution_dot':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution_dot").reshape(batch_size, 1, 14, 14)

    elif args.method == 'taylor_cam':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="taylor_cam").reshape(batch_size, 1, 14, 14)
    elif args.method == 'taylor':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="taylor").reshape(batch_size, 1, 14, 14)
    elif args.method == 'taylor_0.5':
        Res = lrp.generate_LRP(image.cuda()*0.5, start_layer=1, method="taylor").reshape(batch_size, 1, 14, 14)
    elif args.method == 'taylor_0.1':
        Res = lrp.generate_LRP(image.cuda()*0.1, start_layer=1, method="taylor").reshape(batch_size, 1, 14, 14)
    elif args.method == 'taylor_cam_0.5':
        Res = lrp.generate_LRP(image.cuda()*0.5, start_layer=1, method="taylor_cam").reshape(batch_size, 1, 14, 14)

    elif args.method == 'taylor_2nd_order':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="taylor_2nd_order").reshape(batch_size, 1, 14, 14)
    elif args.method == 'taylor_2nd_order_0.5':
        Res = lrp.generate_LRP(image.cuda()*0.5, start_layer=1, method="taylor_2nd_order").reshape(batch_size, 1, 14, 14)
    elif args.method == 'taylor_2nd_order_0.1':
        Res = lrp.generate_LRP(image.cuda()*0.1, start_layer=1, method="taylor_2nd_order").reshape(batch_size, 1, 14, 14)
        
    elif args.method == 'igrad':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="igrad").reshape(batch_size, 1, 14, 14)

    elif args.method == 'igrad2':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="igrad2").reshape(batch_size, 1, 14, 14)

    elif args.method == 'igrad3':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="igrad3").reshape(batch_size, 1, 14, 14)
        
    elif args.method == 't_cam':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="t_cam").reshape(batch_size, 1, 14, 14)    

    elif args.method == 't_cam_diff':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="t_cam_diff").reshape(batch_size, 1, 14, 14)    

    elif args.method == 't_cam_grad':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="t_cam_grad").reshape(batch_size, 1, 14, 14)    
        
    # segmentation test for the partial LRP baseline (last attn layer)
    elif args.method == 'lrp_last_layer':
        Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer", is_ablation=args.is_ablation)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the raw attention baseline (last attn layer)
    elif args.method == 'attn_last_layer':
        Res = orig_lrp.generate_LRP(image.cuda(), method="last_layer_attn", is_ablation=args.is_ablation)\
            .reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the GradCam baseline (last attn layer)
    elif args.method == 'attn_gradcam':
        Res = baselines.generate_cam_attn(image.cuda()).reshape(batch_size, 1, 14, 14)
    elif args.method == 'attn_gradcam_0.5':
        Res = baselines.generate_cam_attn(image.cuda()*0.5).reshape(batch_size, 1, 14, 14)
    elif args.method == 'attn_gradcam_0.1':
        Res = baselines.generate_cam_attn(image.cuda()*0.1).reshape(batch_size, 1, 14, 14)

    elif args.method == 'perturb':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb").reshape(batch_size, 1, 14, 14)
        
#     outputs = []
        
    elif args.method == 'perturb_all':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)
#         Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all").reshape(batch_size, 1, 14, 14)
    elif args.method == 'perturb_none':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method == 'perturb_attn':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method == 'perturb_attn_v':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method.startswith('perturb_all_'):
        start_layer = int(args.method.split('_')[-1])
        Res = lrp.generate_LRP(image.cuda(), start_layer=start_layer, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)

    if args.method not in ['full_lrp' ,'input']:
        # interpolate to full image size (224,224)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear', align_corners=False).cuda()
    original_image = image[0].permute(1, 2, 0).data.cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    if args.method != 'input':
        # threshold between FG and BG is the mean    
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        vis = show_cam_on_image(original_image, Res[0,0].detach().cpu().numpy())
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    else:
        vis = original_image
        vis =  np.uint8(255 * vis)
#         vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    imageio.imsave(os.path.join(args.exp_img_path, 'viz_' + str(index) + '.jpg'), vis)


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):
#     if batch_idx == 10:
#         break
    if args.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()

    eval_batch(images, labels, model, batch_idx)
