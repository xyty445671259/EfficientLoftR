import torch
import cv2
import numpy as np
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter, infer_default_config

long_side = 832
def resize_to_same(img, long_side):
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    new_h, new_w = int(h*scale), int(w*scale)
    # 保证 32 对齐
    new_h, new_w = new_h // 32 * 32, new_w // 32 * 32
    return cv2.resize(img, (new_w, new_h))

# Initialize the matcher with default settings
_default_cfg = deepcopy(infer_default_config)
matcher = LoFTR(config=_default_cfg)
#/home/cxy/gitlab/EfficientLoftR/weights/eloftr_outdoor.ckpt
# Load pretrained weights
matcher.load_state_dict(torch.load("/home/cxy/gitlab/EfficientLoftR/logs/tb_logs/stain_0901_bs=1/version_47/checkpoints/last.ckpt")['state_dict'])
matcher = reparameter(matcher)  # Essential for good performance
matcher = matcher.eval().cuda()

# Load and preprocess images
img0_raw = cv2.imread("/home/cxy/gitlab/EfficientLoftR/data/stain/train/HE_PAS/MP_H_1_1_K2023_20250327_PAS_RRB-21835_S0_S0.png", cv2.IMREAD_COLOR)
img1_raw = cv2.imread("/home/cxy/gitlab/EfficientLoftR/data/stain/train/HE_PAS/MP_H_1_1_K2023_20250327_PASM_RRB-21835_S0_S0.png", cv2.IMREAD_COLOR)

# Resize images to be divisible by 32
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//64*64, img0_raw.shape[0]//64*64))
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//64*64, img1_raw.shape[0]//64*64))

# Convert to grayscale for LoFTR processing
img0_gray = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2GRAY)

# Convert to tensors
img0 = torch.from_numpy(img0_gray)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_gray)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}


# Inference
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # Matched keypoints in image0
    mkpts1 = batch['mkpts1_f'].cpu().numpy()  # Matched keypoints in image1
    mconf = batch['mconf'].cpu().numpy()

print(f"找到 {len(mkpts0)} 个匹配点")
img0_display = img0_raw.copy()  # 直接使用彩色图像
img1_display = img1_raw.copy()  # 直接使用彩色图像
# 创建一个水平拼接的图像用于显示匹配
h1, w1 = img0_display.shape[:2]
h2, w2 = img1_display.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
vis[:h1, :w1] = img0_display
vis[:h2, w1:w1+w2] = img1_display

# 绘制匹配点
for i in range(len(mkpts0)):
    if mconf[i] > 0.3:
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
max_display_size =1024  # 最大显示尺寸
if vis.shape[1] > max_display_size:
    scale = max_display_size / vis.shape[1]
    new_h = int(vis.shape[0] * scale)
    vis = cv2.resize(vis, (max_display_size, new_h))




cv2.imwrite('/home/cxy/gitlab/EfficientLoftR/data/stain/res.jpg', vis)
