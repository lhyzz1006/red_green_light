import os
import cv2

image_dir = './dataset/map/images/train'
label_dir = './dataset/map/labels/train'

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith('.jpg'):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))

    img = cv2.imread(img_path)
    if img is None:
        print(f"[❌] 图像无法读取：{img_path}")
        continue
    h, w = img.shape[:2]

    # 读取标签文件
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'cls {int(cls)}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        print(f"[⚠️] 没有找到对应标签：{label_path}")

    # 创建可缩放窗口并展示全图
    window_name = 'YOLO Label Check'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    key = cv2.waitKey(0)
    if key == 27:  # ESC 退出
        break

cv2.destroyAllWindows()