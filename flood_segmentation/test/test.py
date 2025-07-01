import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from ultralytics import YOLO
import cv2

# ========== 配置部分 ==========
MODEL_PATH   = "best.pt"
IMG_DIR      = "../dataset/test/images"
GT_MASK_DIR  = "../dataset/test/labels"
MASK_EXT     = ".png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# =============================

def evaluate_binary(model, img_dir, gt_dir, mask_ext, device):
    # 初始化混淆矩阵元素
    TP = TN = FP = FN = 0

    img_paths = sorted(sum((glob(os.path.join(img_dir, ext))
                            for ext in ("*.jpg","*.jpeg","*.png")), []))

    for img_p in tqdm(img_paths, desc="Inference"):
        name, _ = os.path.splitext(os.path.basename(img_p))
        gt_p = os.path.join(gt_dir, name + mask_ext)
        if not os.path.isfile(gt_p):
            continue

        # 1) 读取真值掩码，转为二值：0=背景，1=前景
        gt = cv2.imread(gt_p, cv2.IMREAD_UNCHANGED)
        if gt is None: continue
        if gt.ndim == 3: gt = gt[...,0]
        gt = (gt > 0).astype(np.uint8)

        # 2) 模型推理
        res = model.predict(source=img_p, device=device,
                            task="segment", verbose=False)[0]

        H, W = gt.shape
        pred = np.zeros((H, W), dtype=np.uint8)

        # 判空 & resize
        if res.masks is not None and getattr(res.masks, "data", None) is not None and len(res.masks.data):
            masks = res.masks.data.cpu().numpy()  # [N, Mh, Mw]
            for m in masks:
                # 重采样到 (W, H)
                m_rs = cv2.resize(m.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
                pred[m_rs > 0] = 1

        # 3) 累计 TP/TN/FP/FN
        TP += np.logical_and(pred == 1, gt == 1).sum()
        TN += np.logical_and(pred == 0, gt == 0).sum()
        FP += np.logical_and(pred == 1, gt == 0).sum()
        FN += np.logical_and(pred == 0, gt == 1).sum()

    # 4) 计算两类指标
    # 背景类 = 0
    acc_bg = TN / (TN + FP + 1e-12)
    iu_bg  = TN / (TN + FP + FN + 1e-12)
    # 前景类 = 1
    acc_fg = TP / (TP + FN + 1e-12)
    iu_fg  = TP / (TP + FP + FN + 1e-12)

    # 平均
    mPA  = (acc_bg + acc_fg) / 2
    mIoU = (iu_bg  + iu_fg)  / 2

    return {
        "acc_bg": acc_bg,
        "acc_fg": acc_fg,
        "mPA":    mPA,
        "iu_bg":  iu_bg,
        "iu_fg":  iu_fg,
        "mIoU":   mIoU
    }

def main():
    # 加载模型
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    model.model.eval()

    stats = evaluate_binary(model, IMG_DIR, GT_MASK_DIR, MASK_EXT, DEVICE)

    # 打印结果
    print("\n===== Pixel Accuracy =====")
    print(f"Background: {stats['acc_bg']*100:6.2f}%")
    print(f"Foreground: {stats['acc_fg']*100:6.2f}%")
    print(f"Mean (mPA):{stats['mPA']*100:6.2f}%")

    print("\n===== IoU =====")
    print(f"Background: {stats['iu_bg']*100:6.2f}%")
    print(f"Foreground: {stats['iu_fg']*100:6.2f}%")
    print(f"Mean (mIoU):{stats['mIoU']*100:6.2f}%\n")

if __name__ == "__main__":
    main()
