# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""

import numpy as np
from PIL import Image
import cv2
from glob import glob
import os
import torch
import torch.nn.functional as F
from queue import Queue
from tqdm import tqdm
from urllib.request import urlopen

@torch.no_grad()
def eval_davis(model, video_name, videos_path, labels_path, queue_length, topk, size_neighbourhood, model_name, output_dir, patch_size=8):
    color_palette = np.loadtxt(urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"), dtype=np.uint8).reshape(-1, 3)
    frames = read_list_frames(os.path.join(videos_path, video_name))
    labels = read_list_labels(os.path.join(labels_path, video_name))
    
    first_frame, ori_h, ori_w = read_frame(frames[0])
    first_label, _ = read_label(labels[0])
    first_feat, _, _ = encode_frame(model, first_frame)

    queue = Queue(maxsize=queue_length)
    video_output = os.path.join(output_dir, model_name, video_name)
    os.makedirs(video_output, exist_ok=True)

    for frame_path in tqdm(frames[1:], desc="Processing frames"):
        past_feats = [first_feat] + [item[0] for item in queue.queue]
        past_labels = [first_label] + [item[1] for item in queue.queue]

        target_frame = read_frame(frame_path)[0]
        frame_seg, frame_feat = label_propagation(model, past_feats, past_labels, target_frame, topk, size_neighbourhood)

        if queue.full():
            queue.get()
        queue.put((frame_feat, frame_seg))

        frame_seg = F.interpolate(frame_seg, scale_factor=patch_size, mode='bilinear', align_corners=False)[0]
        frame_seg = torch.argmax(norm_mask(frame_seg), dim=0).cpu().numpy().astype(np.uint8)
        frame_seg = np.array(Image.fromarray(frame_seg).resize((ori_w, ori_h), Image.NEAREST))

        output_path = os.path.join(video_output, os.path.basename(frame_path).replace(".jpg", ".png"))
        save_frame_image(output_path, frame_seg, color_palette)

def label_propagation(model, list_past_features, past_labels, frame_tar, topk, size_neighborhood=0):
  feat_tar, h, w = encode_frame(model, frame_tar)
  ncontext = len(list_past_features)

  feats_tar = feat_tar.cuda().unsqueeze(0).repeat(ncontext, 1, 1)
  feats_tar = F.normalize(feats_tar, dim=1, p=2)
  feat_sources = torch.stack(list_past_features).cuda()
  feat_sources = F.normalize(feat_sources, dim=1, p=2)

  aff = torch.exp(torch.bmm(feats_tar, feat_sources.permute(0, 2, 1)) / 0.1)
  if size_neighborhood > 0:
    mask = compute_mask(h, w, size_neighborhood).unsqueeze(0).repeat(ncontext, 1, 1)
    aff *= mask

  aff = aff.transpose(2, 1).reshape(-1, h * w)
  tk_val, _ = torch.topk(aff, dim=0, k=topk)
  tk_val_min, _ = torch.min(tk_val, dim=0)
  aff[aff < tk_val_min] = 0

  aff = aff / torch.sum(aff, keepdim=True, axis=0)

  past_labels = [s.cuda() for s in past_labels]
  segs = torch.cat(past_labels)
  nmb_context, C, h, w = segs.shape
  segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T
  tseg = torch.mm(segs, aff)
  tseg = tseg.reshape(1, C, h, w)

  return tseg, feat_tar

def read_frame(frame_path, patch_size=8, ori_shape=False):
    frame = cv2.imread(frame_path)
    h, w, _ = frame.shape
    new_w, new_h = (int(w // patch_size) * patch_size, h) if ori_shape else (224, 224)
    frame = cv2.resize(frame, (new_w, new_h)).astype(np.float32) / 255.0
    frame = torch.tensor(np.transpose(frame, (2, 0, 1)), dtype=torch.float32)
    return frame, h, w

def read_label(label_path, patch_size=8, ori_shape=False):
  labels = Image.open(label_path)
  w, h = labels.size
  new_w, new_h = (int(w // patch_size) * patch_size, h) if ori_shape else (224, 224)
  resized_labels = labels.resize((new_w // patch_size, new_h // patch_size), 0)
  resized_labels = np.array(resized_labels)
  resized_labels = torch.from_numpy(resized_labels.copy())
  w, h = resized_labels.shape
  n_dims = int(resized_labels.max()+1)
  flattened_labels = resized_labels.type(torch.LongTensor).view(-1,1)
  one_hot = torch.zeros((flattened_labels.shape[0], n_dims)).scatter(1, flattened_labels, 1)
  one_hot = one_hot.view(h, w, n_dims)
  return one_hot.permute(2,0,1).unsqueeze(0), np.asarray(labels) 

def read_list_frames(video_path):
  return sorted([file for file in glob(os.path.join(video_path, '*.jpg'))])

def read_list_labels(labels_path):
  return sorted([file for file in glob(os.path.join(labels_path, '*.png'))])

def norm_mask(mask):
    mask -= mask.min(dim=(0), keepdim=True)[0]
    mask /= mask.max(dim=(0), keepdim=True)[0].clamp(min=1e-6)
    return mask

def compute_mask(h, w, size_neighborhood):
    indices = torch.arange(h * w).view(h, w)
    mask = (indices.unsqueeze(-1) - indices.view(-1))**2 < (size_neighborhood**2)
    return mask.float().cuda(non_blocking=True).reshape(h * w, h * w)

def encode_frame(model, frame, patch_size=8):
    frame = frame.unsqueeze(0).cuda()
    h, w = frame.shape[2] // patch_size, frame.shape[3] // patch_size
    out = model.forward_encoder(frame, 0)[:, 1:, :].squeeze().cuda()
    return out, h, w

def save_frame_image(filename, array, color_palette):
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')