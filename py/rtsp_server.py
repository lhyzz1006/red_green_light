import cv2
from ultralytics import YOLO

from client_sender import send_violation_count

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("rtmp://127.0.0.1/live/stream")

# 定义你画的几条判断线
line_list = [
    ((100, 500), (1000, 500)),  # 比如斑马线前沿
    ((100, 600), (1000, 600)),  # 斑马线中部
]
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not cap.isOpened():
        print("Error opening video stream or file")
        break
    if not ret:
        break

    # 人体检测
    results = model(frame, classes=[0], conf=0.5)  # 只检测人类（class 0）
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    # 画基准线
    for pt1, pt2 in line_list:
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

    # 遍历每个检测到的人
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)  # 脚底位置
        cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)

        # 检查是否跨线（这里假设你想检测跨越 line_list[0]）
        line_y = line_list[0][0][1]  # 获取那条线的 y 值
        if foot_y < line_y:  # 脚底在基准线前 → 未违规
            pass
        else:  # 脚底越线
            count += 1
            cv2.putText(frame, "Violation!", (foot_x, foot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            send_violation_count(count)

    cv2.imshow("Stream", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
