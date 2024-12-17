import os
from eval import get_labels
from tqdm import tqdm
import numpy as np 
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score
from glob import glob


def eval_model(list_videos, videos_path, labels_path, output_dir, model_name):
  scores = []
  for video in tqdm(list_videos):
    list_segs = glob(os.path.join(output_dir, model_name, video, '*.png'))
    list_labels = get_labels(os.path.join(labels_path, video))
    n = len(list_segs)
    print(f"Video {video} with n={n}")
    scores_vid = np.zeros((n-1,2))
    for i in range(1, n):
      pred_seg = Image.open(list_segs[i])
      real_seg = Image.open(list_labels[i])
      pred_seg = np.asarray(pred_seg).reshape(-1)
      real_seg = np.asarray(real_seg).reshape(-1)
      scores_vid[i-1,0] = jaccard_score(real_seg, pred_seg, average='macro')
      scores_vid[i-1,1] = f1_score(real_seg, pred_seg, average='macro')
    scores.append(scores_vid)
  scores = np.concatenate(scores, axis=0)
  scores = np.mean(scores, axis=0)
  print(f'J = {scores[0]}, F = {scores[1]}, J&F = {np.mean(scores)}')
  return scores


