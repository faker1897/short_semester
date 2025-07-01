#!/usr/bin/env python3
"""
test_detection_metrics.py

Evaluate a trained YOLOv8 detection model on a test/validation set
using YOLO-format .txt labels. Computes per-class Accuracy, Precision, Recall, and F1-score.
"""

import os
import numpy as np
import cv2
import time
from ultralytics import YOLO

start = time.perf_counter()

def bbox_iou(box1, box2):
    """Compute IoU between two boxes given as [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-8)

def read_yolo_labels(txt_path, img_w, img_h):
    """
    Read YOLO-format .txt labels and convert to pixel boxes.
    Returns:
      boxes: numpy array of shape (M,4) in [x1,y1,x2,y2]
      classes: numpy array of shape (M,)
    """
    boxes, classes = [], []
    if not os.path.exists(txt_path):
        return np.zeros((0,4)), np.zeros((0,), dtype=int)
    with open(txt_path, 'r') as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(int(cls))
    return np.array(boxes), np.array(classes)

if __name__ == "__main__":
    # --- User-configurable paths ---
    model_path = "runs/train/car_submerge/weights/best.pt"
    img_dir= "dataset/test/images"
    lbl_dir= "dataset/test/labels"
    num_classes = 3  # number of classes (0,1,...)

    # Load model
    model = YOLO(model_path)

    # Initialize per-class counters
    stats = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(num_classes)}

    # Loop over images
    for img_name in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')

        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load ground-truth boxes
        gt_boxes, gt_cls = read_yolo_labels(txt_path, w, h)
        used_gt = set()

        # Run detection
        results = model.predict(source=img_path, task="detect", imgsz=640, verbose=False)[0]
        pred_boxes = results.boxes.xyxy.cpu().numpy()    # shape (N,4)
        pred_cls   = results.boxes.cls.cpu().numpy().astype(int)

        # If no ground-truth, all preds are FP
        if len(gt_boxes) == 0:
            for pc in pred_cls:
                stats[pc]['FP'] += 1
        else:
            # Match predictions to GT
            for pb, pc in zip(pred_boxes, pred_cls):
                # compute IoUs for this pred against all gt of same class
                ious = []
                for i, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
                    if gc != pc:
                        ious.append(0.0)
                    else:
                        ious.append(bbox_iou(pb, gb))
                ious = np.array(ious)
                best_i = ious.argmax() if ious.size > 0 else -1
                if ious.size > 0 and ious.max() >= 0.5 and best_i not in used_gt:
                    stats[pc]['TP'] += 1
                    used_gt.add(best_i)
                else:
                    stats[pc]['FP'] += 1
            # Unmatched GT are FN
            for i, gc in enumerate(gt_cls):
                if i not in used_gt:
                    stats[gc]['FN'] += 1

    # Compute and print per-class metrics
    for c in range(num_classes):
        tp = stats[c]['TP']
        fp = stats[c]['FP']
        fn = stats[c]['FN']
        # Accuracy: TP / (TP + FP + FN)
        accuracy = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"Class {c}:")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}\n")

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

total_tp = sum(stats[c]['TP'] for c in range(num_classes))
total_fp = sum(stats[c]['FP'] for c in range(num_classes))
total_fn = sum(stats[c]['FN'] for c in range(num_classes))

overall_accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-8)
overall_precision = total_tp / (total_tp + total_fp + 1e-8)
overall_recall    = total_tp / (total_tp + total_fn + 1e-8)
overall_f1        = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-8)

print("Overall metrics:")
print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
print(f"  Accuracy:  {overall_accuracy:.4f}")
print(f"  Precision: {overall_precision:.4f}")
print(f"  Recall:    {overall_recall:.4f}")
print(f"  F1-score:  {overall_f1:.4f}")
