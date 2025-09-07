from ultralytics import YOLO
import cv2

# 1. 加载模型
model = YOLO("model/best.pt")  # 改成你的模型路径

# 2. 加载测试图片
img_path = "F:/signal/code/dataset/map/images/train/IMG_5275.jpg"  # 替换为你要测试的图片路径
# ✅ 读取原图（整张）
img = cv2.imread(img_path)
assert img is not None, f"图片读取失败：{img_path}"

# ✅ 推理
results = model.predict(img, conf=0.2, imgsz=img.shape[:2][::-1])[0]  # 保持原始尺寸

# ✅ 绘制检测框
print("预测框数量：", len(results.boxes))
for box in results.boxes:
    print("类别：", model.names[int(box.cls[0])], "置信度：", float(box.conf[0]))

    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ✅ 展示整张图像
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)  # 允许手动缩放窗口
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()