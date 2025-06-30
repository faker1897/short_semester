import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from ultralytics import YOLO
from pathlib import Path

def txt_to_mask(txt_path, image_size):
    """
    将 YOLOv8 segmentation 的 .txt 标注（class + 多边形顶点）
    转成二值的 Flood 掩码（uint8 0/1）。
    假设 .txt 每行：cls x1 y1 x2 y2 ... xN yN （归一化坐标）
    """
    W, H = image_size
    mask = np.zeros((H, W), dtype=np.uint8)
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(float(parts[0]))   # 应为 0（flood）
            coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
            poly = [(x * W, y * H) for x, y in coords]
            m = Image.new('L', (W, H), 0)
            ImageDraw.Draw(m).polygon(poly, outline=1, fill=1)
            mask[np.array(m) == 1] = 1
    return mask

def main():
    # 1) 读取你自己的 data.yaml（脚本同目录下）
    #    要确保其中有：
    #      test:   "./dataset/test/images"
    #      labels: "./dataset/test/labels"
    #      nc: 1
    #      names: ["flood"]
    cfg = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    img_dir   = Path(cfg["test"])
    label_dir = Path(cfg["labels"])
    names     = cfg["names"]
    assert len(names) == 1, "本脚本仅支持单类 flood 评估"

    # 2) 扫描所有测试图片
    img_paths = sorted(img_dir.glob("*.*"))
    if not img_paths:
        raise RuntimeError(f"在 {img_dir} 下未找到任何图片")

    # 3) 加载训练好的模型权重
    model = YOLO("best.pt")

    # 4) 用于统计 Flood 类像素级 TP/FP/FN
    tp = fp = fn = 0

    # 5) 遍历每张图，预测 & 累加
    for img_p in tqdm(img_paths, desc="Evaluating flood"):
        # 5.1 预测分割实例
        res = model.predict(source=str(img_p), task="segment", conf=0.1, verbose=False)[0]
        masks = res.masks.data.cpu().numpy()   # (N, H, W)

        # 5.2 构建二值预测掩码 pred (1 表示 flood)
        if masks.shape[0] == 0:
            pred = np.zeros((res.orig_shape[0], res.orig_shape[1]), dtype=np.uint8)
        else:
            # masks 可能是 bool 或 0/1，先强制为 bool
            bool_masks = masks.astype(bool)
            H, W = bool_masks.shape[1:]
            pred = np.zeros((H, W), dtype=np.uint8)
            for inst_mask in bool_masks:
                pred[inst_mask] = 1

        # 5.3 读取并解析 GT
        txt_p = label_dir / f"{img_p.stem}.txt"
        if not txt_p.exists():
            raise FileNotFoundError(f"找不到标签文件：{txt_p}")
        # 注意：txt_to_mask 返回 0/1，1 表示 flood
        gt = txt_to_mask(txt_p, (pred.shape[1], pred.shape[0]))

        # 5.4 累加 TP/FP/FN（像素级）
        tp += np.logical_and(pred == 1, gt == 1).sum()
        fp += np.logical_and(pred == 1, gt == 0).sum()
        fn += np.logical_and(pred == 0, gt == 1).sum()

    # 6) 计算单类 mPA（PA）和 mIoU
    mPA  = tp / (tp + fn + 1e-12)
    mIoU = tp / (tp + fp + fn + 1e-12)

    print(f"\nmPA (Mean Pixel Accuracy): {mPA*100:.2f}%")
    print(f"mIoU (Mean IoU)          : {mIoU*100:.2f}%")

if __name__ == "__main__":
    main()
