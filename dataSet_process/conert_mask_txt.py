
import os
import cv2


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            polygons.append(cnt.squeeze())
    return polygons

def convert_masks_to_yolo(mask_dir, save_dir, image_dir, class_id=0):
    os.makedirs(save_dir, exist_ok=True)
    for mask_name in os.listdir(mask_dir):
        if not mask_name.endswith('.png'):
            continue

        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        image_path = os.path.join(image_dir, mask_name)
        if not os.path.exists(image_path):
            image_path = image_path.replace('.png', '.jpg')
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        polygons = mask_to_polygons(binary)
        lines = []

        for polygon in polygons:
            if len(polygon.shape) != 2:
                continue
            normalized = [(x / w, y / h) for x, y in polygon]
            flat = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized])
            line = f"{class_id} {flat}"
            lines.append(line)

        txt_name = mask_name.replace('.png', '.txt')
        with open(os.path.join(save_dir, txt_name), 'w') as f:
            f.write('\n'.join(lines))

    print(f"✅ {mask_dir} → {save_dir}")


convert_masks_to_yolo(
    mask_dir='masks',
    save_dir='labels',
    image_dir='images'
)

