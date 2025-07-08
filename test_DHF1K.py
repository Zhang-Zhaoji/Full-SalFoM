import sys
import os
import numpy as np
import cv2
import torch
# from model import VideoSaliencyModel
from Model import net as VideoSaliencyModel
from scipy.ndimage.filters import gaussian_filter
from loss import kldiv, cc, nss
import argparse
from torch.utils.data import DataLoader
from dataloader import DHF1KDataset
from utils1 import *
import time
from tqdm import tqdm
from torchvision import transforms, utils
from os.path import join
import copy
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('USE GPU 0')


def validate(args):
    path_indata = args.path_indata
    file_weight = args.file_weight
    len_temporal = args.clip_size

    model = VideoSaliencyModel()

    model.load_state_dict(torch.load(file_weight))
    # model = model.cuda()
    model = model.cpu()
    model.eval()

    list_indata = [d for d in os.listdir(path_indata) if os.path.isdir(os.path.join(path_indata, d))]
    list_indata.sort()

    if args.start_idx != -1:
        _len = (1.0 / float(args.num_parts)) * len(list_indata)
        list_indata = list_indata[int((args.start_idx - 1) * _len): int(args.start_idx * _len)]
    for dname in tqdm.tqdm(list_indata):
        print('processing ' + dname, flush=True)
        # list_frames = [f for f in os.listdir(os.path.join(path_indata, dname, 'images')) if
        #                os.path.isfile(os.path.join(path_indata, dname, 'images', f))]
        list_frames = [f for f in os.listdir(os.path.join(path_indata, dname )) if
                       os.path.isfile(os.path.join(path_indata, dname, f))]
        list_frames.sort()
        os.makedirs(join(args.save_path, dname), exist_ok=True)

        temp = [list_frames[0] for _ in range(15)]
        temp.extend(list_frames)
        list_frames = copy.deepcopy(temp)

        snippet = []
        for i in tqdm.trange(len(list_frames)):
            # torch_img, img_size = torch_transform(os.path.join(path_indata, dname, 'images', list_frames[i]))
            torch_img, img_size = torch_transform(os.path.join(path_indata, dname, list_frames[i]))

            snippet.append(torch_img)
            

            if i >= len_temporal - 1:
                clip = torch.FloatTensor(torch.stack(snippet, dim=0)).unsqueeze(0)
                clip = clip.permute((0, 2, 1, 3, 4))
                del snippet[0]
                process(model, clip, dname, list_frames[i], args, img_size, [],[],[],[],[])

    sys.stdout.flush()


def torch_transform(path):
    img_transform = transforms.Compose([
        # transforms.Resize((224, 384)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return torch.FloatTensor(bl)


def process(model, clip, dname, frame_no, args, img_size, cc_loss, kldiv_loss, nss_loss, sim_loss, auc_j_loss):
    with torch.no_grad():
        smap = model(clip.cpu())

    smap = smap.cpu().data[0].numpy()
    smap = cv2.resize(smap, (img_size[0], img_size[1]))
    smap = blur(smap)
    # 定性分析
    img_save(smap, join(args.save_path, dname, frame_no), normalize=True)
    # 定量分析
    # gt = np.array(Image.open(join(args.path_indata, dname, 'maps', frame_no)).convert('L'))
    # gt = gt.astype('float')
    # if np.max(gt) > 1.0:
    #     gt = gt / 255.0
    # labels = torch.FloatTensor(gt)
    # labels = labels.cpu()

    # fixations = np.array(Image.open(join(args.path_indata, dname, 'fixation', frame_no)).convert('L'))
    # fixations = fixations.astype('float')
    # fixations = (fixations > 0.5).astype('float')
    # fixations = torch.FloatTensor(fixations)
    # fixations = fixations.cpu()

    # smap = smap.cpu()
    # smap = smap.unsqueeze(0)
    # labels = labels.unsqueeze(0)
    # fixations = fixations.unsqueeze(0)

    # cc_loss.update(cc(smap, labels))
    # kldiv_loss.update(kldiv(smap, labels))
    # nss_loss.update(nss(smap, fixations))
    # sim_loss.update(similarity(smap, labels))
    # auc_j_loss.update(auc_judd(smap, fixations))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_weight', default="./SalFoM_DHF1K.pth", type=str)
    parser.add_argument('--save_path', default='./results', type=str)
    parser.add_argument('--start_idx', default=-1, type=int)
    parser.add_argument('--num_parts', default=4, type=int)
    parser.add_argument('--path_indata', default=r'E:\Li Lab\itti_and_lif\DLexperiments\TASED\TASED-Net\example', type=str)
    parser.add_argument('--clip_size', default=16, type=int)
    args = parser.parse_args()

    validate(args)
