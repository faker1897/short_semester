import os
import numpy as np
from glob import glob

import torch
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# ========== 配置区 ==========
import os
# 允许重复加载 OpenMP 运行时，避免 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH   = "best.pt"
IMG_DIR      = "../dataset/test/images"
GT_MASK_DIR  = "../dataset/test/labels"
MASK_EXT     = ".png"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CM_OUTPUT    = "confusion_matrix.png"
# ============================

def evaluate_and_plot_cm(model, img_dir, gt_dir, mask_ext, device, cm_output):
    # 初始化四元组
    TP = TN = FP = FN = 0

    # 收集测试图片
    img_paths = sorted(sum((glob(os.path.join(img_dir, ext))
                            for ext in ("*.jpg","*.jpeg","*.png")), []))

    for img_p in tqdm(img_paths, desc="Inference & Eval"):
        name, _ = os.path.splitext(os.path.basename(img_p))
        gt_p = os.path.join(gt_dir, name + mask_ext)
        if not os.path.isfile(gt_p):
            print(f"[跳过] 未找到真值掩码：{gt_p}")
            continue

        # 1) 读取真值掩码，二值化
        gt = cv2.imread(gt_p, cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        if gt.ndim == 3:
            gt = gt[...,0]
        gt = (gt > 0).astype(np.uint8)
        H, W = gt.shape

        # 2) 模型推理
        res = model.predict(source=img_p, device=device,
                            task="segment", verbose=False)[0]

        # 3) 构造预测掩码
        pred = np.zeros((H, W), dtype=np.uint8)
        if res.masks is not None and getattr(res.masks, "data", None) is not None and len(res.masks.data):
            masks = res.masks.data.cpu().numpy()  # [N, Mh, Mw]
            for m in masks:
                m_rs = cv2.resize(m.astype(np.uint8), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
                pred[m_rs > 0] = 1

        # 4) 累积 TP/TN/FP/FN
        TP += np.logical_and(pred == 1, gt == 1).sum()
        TN += np.logical_and(pred == 0, gt == 0).sum()
        FP += np.logical_and(pred == 1, gt == 0).sum()
        FN += np.logical_and(pred == 0, gt == 1).sum()

    # 原始混淆矩阵
    cm = np.array([[TN, FP],
                   [FN, TP]])

    # 打印原始混淆矩阵
    print("Original Confusion Matrix:")
    print(cm)

    # 归一化混淆矩阵（按行）
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(np.float32) / row_sums
    print("\nRow-wise Normalized Confusion Matrix:")
    print(np.round(cm_norm, 4))

    # 绘制原始和归一化混淆矩阵
    labels = ["Background", "Foreground"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, matrix, title, annotate_vals in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix", "Normalized Confusion Matrix"],
        [False, True]
    ):
        im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(title)
        ax.set_ylabel('True Label') if ax is axes[0] else None
        ax.set_xlabel('Predicted Label')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        thresh = matrix.max() / 2
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                txt = f"{val:.2f}" if annotate_vals else f"{int(val)}"
                ax.text(j, i, txt,
                        ha="center", va="center",
                        color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(cm_output, dpi=150)
    print(f"已保存混淆矩阵图像到: {cm_output}")

    # 计算并打印 Pixel Accuracy 和 IoU
    acc_bg = TN / (TN + FP + 1e-12)
    acc_fg = TP / (TP + FN + 1e-12)
    mPA    = (acc_bg + acc_fg) / 2
    iu_bg  = TN / (TN + FP + FN + 1e-12)
    iu_fg  = TP / (TP + FP + FN + 1e-12)
    mIoU   = (iu_bg + iu_fg) / 2

    print("\n===== Pixel Accuracy =====")
    print(f" Background: {acc_bg*100:6.2f}%")
    print(f" Foreground: {acc_fg*100:6.2f}%")
    print(f" Mean (mPA):{mPA*100:6.2f}%")

    print("\n===== IoU =====")
    print(f" Background: {iu_bg*100:6.2f}%")
    print(f" Foreground: {iu_fg*100:6.2f}%")
    print(f" Mean (mIoU):{mIoU*100:6.2f}%")

def main():
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    model.model.eval()
    evaluate_and_plot_cm(model, IMG_DIR, GT_MASK_DIR, MASK_EXT, DEVICE, CM_OUTPUT)

if __name__ == "__main__":
    main()
