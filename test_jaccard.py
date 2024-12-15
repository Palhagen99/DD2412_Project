from sklearn.metrics import jaccard_score
from PIL import Image
import numpy as np

#real_seg = Image.open("./data/DAVIS/Annotations/480p/bear/00001.png")
real_seg = Image.open("./output/davis/test2/bear/00081.png")
pred_seg = Image.open("./output/davis/test1/bear/00001.png")
pred_seg = np.asarray(pred_seg).reshape(-1)
real_seg = np.asarray(real_seg).reshape(-1)

print(jaccard_score(real_seg, pred_seg, average='macro'))