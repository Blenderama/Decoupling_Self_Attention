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

import torch.nn.functional as F
import pdb

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

args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)
args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
if not os.path.exists(args.exp_np_path):
    os.makedirs(args.exp_np_path)

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

# LRP
model_LRP = vit_LRP(pretrained=True).cuda()
model_LRP.eval()
lrp = LRP(model_LRP)

if args.method in ['lrp_last_layer']:
    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)

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
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))

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
    
    # segmentation test for the LRP baseline (this is full LRP, not partial)
    elif args.method == 'full_lrp':
        Res = orig_lrp.generate_LRP(image.cuda(), method="full").reshape(batch_size, 1, 224, 224)
    
    # segmentation test for our method
    elif args.method == 'transformer_attribution':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution").reshape(batch_size, 1, 14, 14)
    elif args.method == 'transformer_attribution_0':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="transformer_attribution").reshape(batch_size, 1, 14, 14)
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
    elif args.method == 'perturb_patch':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_patch", index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method == 'perturb_patch_all':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_patch_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method == 'perturb_attn':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method == 'perturb_attn_v':
        Res = lrp.generate_LRP(image.cuda(), start_layer=0, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)
    elif args.method.startswith('perturb_all_'):
        start_layer = int(args.method.split('_')[-1])
        Res = lrp.generate_LRP(image.cuda(), start_layer=start_layer, method="perturb_all", ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks], index=cls_index).reshape(batch_size, 1, 14, 14)

    if args.method != 'full_lrp':
        # interpolate to full image size (224,224)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear', align_corners=False).cuda()
    
    # threshold between FG and BG is the mean    
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0
    
    ret = Res.mean()
#     ret = Res.max()*0.2

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0


    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear', align_corners=True)
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear', align_corners=True)
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        # relevance = Res[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)
    # TEST
    if Res.max() == 0:
        pred = Res.clamp(min=args.thr)
    else:
        pred = Res.clamp(min=args.thr) / (Res.max())
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):
    if batch_idx > 10:
        pdb.set_trace()
    if args.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()
    # print("image", image.shape)
    # print("lables", labels.shape)

    correct, labeled, inter, union, ap, f1, pred, target = eval_batch(images, labels, model, batch_idx)

    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)
    iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))
#     print(IoU)
predictions = np.concatenate(predictions)
targets = np.concatenate(targets)

pr, rc, thr = precision_recall_curve(targets, predictions)
np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

plt.figure()
plt.plot(rc, pr)
plt.savefig(os.path.join(saver.experiment_dir, 'PR_curve_{}.png'.format(args.method)))

txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
# txtfile = 'result_mIoU_%.4f.txt' % mIoU
fh = open(txtfile, 'w')
print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
fh.close()
