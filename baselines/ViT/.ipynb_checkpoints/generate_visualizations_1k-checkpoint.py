import os
from tqdm import tqdm
import h5py

import argparse

# Import saliency methods and models
from misc_functions import *

from ViT_explanation_generator import Baselines, LRP
from ViT_new import vit_base_patch16_224
from ViT_LRP import vit_base_patch16_224 as vit_LRP
from ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP

from torchvision.datasets import ImageNet
import pdb

def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(args):
    first = True
    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            if batch_idx > 10:
                break
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            data.requires_grad_()

            index = None
            if args.vis_class == 'target':
                index = target

            output = lrp.model(data)
            cls_index = np.argmax(output.cpu().data.numpy(), axis=-1)
            lrp.model.zero_grad()
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, cls_index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)
            one_hot.backward(retain_graph=True)
            
            if args.method == 'rollout':
                Res = baselines.generate_rollout(data, start_layer=1).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'lrp':
                Res = lrp.generate_LRP(data, start_layer=1, index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'transformer_attribution':
                Res = lrp.generate_LRP(data, start_layer=1, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'transformer_attribution_0':
                Res = lrp.generate_LRP(data, start_layer=0, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'sas':
                Res = lrp.generate_LRP(data, start_layer=1, method="sas", index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'transformer_attribution_0.5':
                Res = lrp.generate_LRP(data*0.5, start_layer=1, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'transformer_attribution_0.1':
                Res = lrp.generate_LRP(data*0.1, start_layer=1, method="grad", index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'transformer_attribution_last':
                Res = lrp.generate_LRP(data, start_layer=1, method="transformer_attribution_last", index=index).reshape(data.shape[0], 1, 14, 14)
                
            elif args.method == 'igrad':
                Res = lrp.generate_LRP(data, start_layer=1, method="igrad", index=index).reshape(data.shape[0], 1, 14, 14)
#                 Res = Res_all.reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'transformer_attribution_dot':
                Res = lrp.generate_LRP(data, start_layer=1, method="transformer_attribution_dot", index=index).reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 't_cam':
                Res = lrp.generate_LRP(data, start_layer=1, method="t_cam", index=index).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'full_lrp':
                Res = orig_lrp.generate_LRP(data, method="full", index=index).reshape(data.shape[0], 1, 224, 224)
                # Res = Res - Res.mean()

            elif args.method == 'lrp_last_layer':
                Res = orig_lrp.generate_LRP(data, method="last_layer", is_ablation=args.is_ablation, index=index) \
                    .reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'attn_last_layer':
                Res = lrp.generate_LRP(data, method="last_layer_attn", is_ablation=args.is_ablation) \
                    .reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'attn_gradcam':
                Res = baselines.generate_cam_attn(data, index=index).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'attn_gradcam_0.1':
                Res = baselines.generate_cam_attn(data*0.1, index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'attn_gradcam_0.5':
                Res = baselines.generate_cam_attn(data*0.5, index=index).reshape(data.shape[0], 1, 14, 14)
                
            elif args.method == 'perturb':
                Res = lrp.generate_LRP(data, start_layer=0, method="perturb", index=index).reshape(data.shape[0], 1, 14, 14)
                
#             outputs = []
                
            elif args.method == 'perturb_all_0':
                Res = lrp.generate_LRP(data, start_layer=0, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_none':
                Res = lrp.generate_LRP(data, start_layer=0, method="perturb_all", index=cls_index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_attn':
                Res = lrp.generate_LRP(data, start_layer=0, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_attn_v':
                Res = lrp.generate_LRP(data, start_layer=0, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_all_1':
                Res = lrp.generate_LRP(data, start_layer=1, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_all_2':
                Res = lrp.generate_LRP(data, start_layer=2, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_all_3':
                Res = lrp.generate_LRP(data, start_layer=3, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'perturb_all_4':
                Res = lrp.generate_LRP(data, start_layer=4, method="perturb_all", index=cls_index, ext_attn_list=[blk.attn.get_attn() for blk in lrp.model.blocks], ext_v_list = [blk.attn.get_v() for blk in lrp.model.blocks], ext_y_list=[blk.get_y() for blk in lrp.model.blocks]).reshape(data.shape[0], 1, 14, 14)
            
            elif args.method == 'taylor':
                Res = lrp.generate_LRP(data, start_layer=1, method="taylor", index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'taylor_cam':
                Res = lrp.generate_LRP(data, start_layer=1, method="taylor_cam", index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'taylor_cam_0.5':
                Res = lrp.generate_LRP(data, start_layer=1, method="taylor_cam_0.5", index=index).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'taylor_2nd_order':
                Res = lrp.generate_LRP(data, start_layer=1, method="taylor_2nd_order", index=index).reshape(data.shape[0], 1, 14, 14)
            elif args.method == 'taylor_2nd_order_0.1':
                Res = lrp.generate_LRP(data*0.1, start_layer=1, method="taylor_2nd_order", index=index).reshape(data.shape[0], 1, 14, 14)

            elif args.method == 'taylor_2nd_order_0.5':
                Res = lrp.generate_LRP(data, start_layer=1, method="taylor_2nd_order_0.5", index=index).reshape(data.shape[0], 1, 14, 14)

            if args.method != 'full_lrp' and args.method != 'input_grads':
                Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
#             print(Res)
            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
#                         choices=['rollout', 'lrp', 'transformer_attribution', 'transformer_attribution_last', 'transformer_attribution_dot', 'full_lrp', 'lrp_last_layer', 'igrad', 'attn_last_layer', 'attn_gradcam', 't_cam', 'perturb', 'perturb_all', 'taylor', 'taylor_2nd_order','taylor_2nd_order_0.5', 'taylor_2nd_order_0.1', 'taylor_cam', 'taylor_cam_0.5', 'attn_gradcam_0.1', 'attn_gradcam_0.5'],
                        help='')
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
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
    parser.add_argument('--imagenet-validation-path', type=str,
                        required=True,
                        help='')
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    model = vit_base_patch16_224(pretrained=True).cuda()
    baselines = Baselines(model)

    # LRP
    model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)

    # orig LRP
    model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    compute_saliency_and_save(args)
