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

from ViT_explanation_generator_swin import LRP
from ViT_LRP_swin import swin_transformer as vit_LRP

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
                    default='swin',
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

    # LRP
model_LRP = vit_LRP(pretrained=True).cuda()
model_LRP.eval()
lrp = LRP(model_LRP)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

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
    
    output = lrp.model(image.cuda())
    cls_index = np.argmax(output.cpu().data.numpy(), axis=-1)
    
    Res = torch.zeros([1,1,56,56]).cuda()
    for i in tqdm(range(8)):
        for j in range(8):
            Res_ = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", index=cls_index, k=i*8+j)
            Res[..., i::8, j::8] += Res_.view(7,7)
    Res = torch.nn.functional.interpolate(Res, scale_factor=4, mode='bilinear', align_corners=False).cuda()
    
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

    eval_batch(images, labels, model_LRP, batch_idx)
