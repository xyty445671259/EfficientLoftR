import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict

def generate_wsi_npz(image_root, annotation_dir, output_path, min_matches=5, max_pairs=1000):
    """
    为多染色 WSI 图像生成 NPZ 文件 (无场景区分)
    :param image_root: WSI 图像根目录
    :param annotation_dir: LabelMatch 标注文件目录
    :param output_path: 输出 NPZ 文件路径
    :param min_matches: 每对图像的最小匹配点数
    :param max_pairs: 最大图像对数
    """
    # 1. 收集所有图像路径
    image_paths = []
    for stain_type in os.listdir(image_root):
        stain_dir = osp.join(image_root, stain_type)
        if not osp.isdir(stain_dir):
            continue
            
        for img_name in os.listdir(stain_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.tiff')):
                img_path = osp.join(stain_type, img_name)
                image_paths.append(img_path)
    
    print(f"Found {len(image_paths)} images from {len(os.listdir(image_root))} stain types")
    
    # 2. 按玻片ID分组 (可选)
    slide_images = defaultdict(list)
    for img_path in image_paths:
        # 假设文件名格式: SlideID_RegionID_Stain.png
        slide_id = img_path.split('_')[0]
        slide_images[slide_id].append(img_path)
    
    # 3. 加载 LabelMatch 标注
    all_pairs = []
    for anno_file in os.listdir(annotation_dir):
        if not anno_file.endswith('.json'):
            continue
            
        with open(osp.join(annotation_dir, anno_file), 'r') as f:
            try:
                annotations = json.load(f)
                all_pairs.extend(annotations.get('pairs', []))
            except json.JSONDecodeError:
                print(f"Error reading {anno_file}, skipping")
    
    # 4. 处理标注并创建匹配对
    pair_names = []  # 存储 [img0_path, img1_path]
    all_matches = []  # 存储匹配点 [[x0,y0,x1,y1], ...]
    
    # 用于验证图像是否存在
    valid_images = set(image_paths)
    
    # 处理手动标注的对
    for pair in tqdm(all_pairs, desc="Processing manual pairs"):
        img0 = pair['image0'].replace('\\', '/')  # 统一路径格式
        img1 = pair['image1'].replace('\\', '/')
        
        # 验证图像存在
        if img0 in valid_images and img1 in valid_images:
            matches = np.array(pair['matches'])
            
            # 检查匹配点数量
            if len(matches) >= min_matches:
                pair_names.append([img0, img1])
                all_matches.append(matches)
    
    print(f"Loaded {len(pair_names)} manual pairs")
    
    # 5. 自动创建补充配对 (不同染色类型间)
    stain_types = sorted(os.listdir(image_root))
    stain_combinations = []
    
    # 生成所有可能的染色组合
    for i in range(len(stain_types)):
        for j in range(i+1, len(stain_types)):
            stain_combinations.append((stain_types[i], stain_types[j]))
    
    # 按玻片自动配对
    for slide_id, slide_imgs in tqdm(slide_images.items(), desc="Generating auto pairs"):
        # 按染色类型分组
        stain_groups = defaultdict(list)
        for img_path in slide_imgs:
            stain_type = img_path.split(osp.sep)[0]
            stain_groups[stain_type].append(img_path)
        
        # 为每种染色组合创建配对
        for stain_a, stain_b in stain_combinations:
            imgs_a = stain_groups.get(stain_a, [])
            imgs_b = stain_groups.get(stain_b, [])
            
            if not imgs_a or not imgs_b:
                continue
                
            # 创建所有可能的配对
            for img_a in imgs_a:
                for img_b in imgs_b:
                    # 避免重复配对
                    if [img_a, img_b] in pair_names or [img_b, img_a] in pair_names:
                        continue
                        
                    pair_names.append([img_a, img_b])
                    all_matches.append(np.zeros((min_matches, 4)))  # 占位匹配点
    
    # 6. 限制总对数
    if len(pair_names) > max_pairs:
        pair_names = pair_names[:max_pairs]
        all_matches = all_matches[:max_pairs]
    
    # 7. 转换为 NumPy 数组
    pair_names = np.array(pair_names, dtype=object)
    all_matches = np.array(all_matches, dtype=object)
    
    # 设置重叠分数 (手动标注为1.0，自动配对为0.8)
    overlap_scores = np.ones(len(pair_names), dtype=np.float32)
    overlap_scores[len(all_pairs):] = 0.8  # 自动配对的分数
    
    # 8. 保存为 NPZ 文件
    np.savez(
        output_path,
        name=pair_names,
        matches=all_matches,
        overlap_scores=overlap_scores
    )
    
    print(f"Saved {len(pair_names)} pairs to {output_path}")
    print(f"Statistics:")
    print(f"- Manual pairs: {len(all_pairs)}")
    print(f"- Auto pairs: {len(pair_names) - len(all_pairs)}")
    print(f"- Min matches per pair: {min(len(m) for m in all_matches)}")
    print(f"- Max matches per pair: {max(len(m) for m in all_matches)}")
    print(f"- Average matches: {sum(len(m) for m in all_matches) / len(all_matches):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate NPZ files for multi-stain WSI dataset')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Root directory of WSI images organized by stain type')
    parser.add_argument('--annotation_dir', type=str, required=True,
                        help='Directory containing LabelMatch JSON annotations')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output NPZ file path')
    parser.add_argument('--min_matches', type=int, default=10,
                        help='Minimum number of matches per pair')
    parser.add_argument('--max_pairs', type=int, default=1000,
                        help='Maximum total pairs')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(osp.dirname(args.output_path), exist_ok=True)
    
    generate_wsi_npz(
        image_root=args.image_root,
        annotation_dir=args.annotation_dir,
        output_path=args.output_path,
        min_matches=args.min_matches,
        max_pairs=args.max_pairs
    )