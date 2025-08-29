import torch
import cv2
import numpy as np
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter

# Initialize the matcher with default settings
_default_cfg = deepcopy(full_default_cfg)
matcher = LoFTR(config=_default_cfg)

# Load pretrained weights
matcher.load_state_dict(torch.load("/home/cxy/medipath/learning/EfficientLoFTR/logs/tb_logs/stain_1024_bs=1/version_0/checkpoints/last.ckpt")['state_dict'])
matcher = reparameter(matcher)  # Essential for good performance
matcher = matcher.eval().cuda()

# Load and preprocess images
img0_raw = cv2.imread("/home/cxy/medipath/learning/EfficientLoFTR/data/stain/img/K2023-0122_HE_S0.png", cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread("/home/cxy/medipath/learning/EfficientLoFTR/data/stain/img/K2023-0122_PAS_S0.png", cv2.IMREAD_GRAYSCALE)

# Resize images to be divisible by 32
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

# Convert to tensors
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # Matched keypoints in image0
    mkpts1 = batch['mkpts1_f'].cpu().numpy()  # Matched keypoints in image1
    mconf = batch['mconf'].cpu().numpy()

print(f"找到 {len(mkpts0)} 个匹配点")
img0_display = cv2.cvtColor(img0_raw, cv2.COLOR_GRAY2BGR)
img1_display = cv2.cvtColor(img1_raw, cv2.COLOR_GRAY2BGR)
# 创建一个水平拼接的图像用于显示匹配
h1, w1 = img0_display.shape[:2]
h2, w2 = img1_display.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
vis[:h1, :w1] = img0_display
vis[:h2, w1:w1+w2] = img1_display

# 绘制匹配点
for i in range(len(mkpts0)):
    if mconf[i] > 0.7:
        # 随机颜色，但为了更好的可视化，可以使用基于置信度的颜色
        color = np.random.randint(0, 255, 3).tolist()
        
        # 在第一张图像上绘制点
        pt0 = (int(round(mkpts0[i][0])), int(round(mkpts0[i][1])))
        cv2.circle(vis, pt0, 3, color, -1)
        
        # 在第二张图像上绘制点（注意x坐标需要偏移）
        pt1 = (int(round(mkpts1[i][0])) + w1, int(round(mkpts1[i][1])))
        cv2.circle(vis, pt1, 3, color, -1)
        
        # 绘制连接线
        cv2.line(vis, pt0, pt1, color, 1)

# 调整图像大小以便显示（如果太大）
max_display_size = 2048  # 最大显示尺寸
if vis.shape[1] > max_display_size:
    scale = max_display_size / vis.shape[1]
    new_h = int(vis.shape[0] * scale)
    vis = cv2.resize(vis, (max_display_size, new_h))




cv2.imwrite('1.jpg', vis)
