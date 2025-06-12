import csv
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import threading
import queue

from client_sender import send_violation_count

# 配置参数
VIDEO_PATH = 'rtmp://127.0.0.1/live/stream'

# 初始化模型 
yolo_model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)

# 帧缓冲区，最多保留2帧（防止延迟累积）
frame_buffer = queue.Queue(maxsize=2)
stop_flag = False
def frame_reader(cap):
    global stop_flag
    while not stop_flag and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_buffer.full():
            frame_buffer.put(frame)

# 判断当前帧红灯是否亮
def is_red_light_on(frame):
    return True

def is_inside_red_area(x, y, red_mask):
    height, width = red_mask.shape
    if 0 <= x < width and 0 <= y < height:
        return red_mask[y, x] > 0
    return False

# 存储每个行人的历史脚部坐标和时间
track_history = {}
violation_records = []
red_area_ids = set()

CONFIDENCE_THRESHOLD = 0.6
# 主处理函数
def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    #输出结果视频，暂不需要
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # out = cv2.VideoWriter('result_cw.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    violation_log = set()
    cross_log = set()
    if not cap.isOpened():
            print("❌ 无法打开 RTMP 流")
            return
    reader_thread = threading.Thread(target=frame_reader, args=(cap,), daemon=True)
    reader_thread.start()
    while True:
        if frame_buffer.empty():
            continue  # 没帧可处理，跳过
        frame = frame_buffer.get()

        current_time = time.time()
        red_light = is_red_light_on(frame)
        results = yolo_model(frame)[0]
        
        # 实时检测红色区域（动态不规则轮廓）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 只保留面积较大的轮廓
        red_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 400]
        for cnt in red_contours:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue  # 忽略这个框
                if 1.0 * (y2 - y1) / (x2 - x1) < 1:  # 宽高比阈值
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        index = 0
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            box_height = max(1 * int(w - l), int(h - t))
            box_width = max(1, int(w - l))

            # 避免半身误判：将脚部估计放到框底向上10%
            foot_x = int(l + box_width / 2)
            foot_y = int(t + box_height * 0.95)  # 上移一些
            cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 255), -1)

            # 保存历史坐标
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(((foot_x, foot_y), current_time))
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

            # 判断移动速度（像素/秒）
            foot_speed = 0
            if len(track_history[track_id]) >= 2:
                (x0, y0), t0 = track_history[track_id][0]
                (x1, y1), t1 = track_history[track_id][-1]
                dist = np.linalg.norm(np.array([x1, y1]) - np.array([x0, y0]))
                dt = t1 - t0
                if dt > 0:
                    foot_speed = dist / dt
            in_red = is_inside_red_area(foot_x, foot_y, red_mask)
            if red_light:
                if in_red and track_id not in red_area_ids:
                    red_area_ids.add(track_id)  # 记录首次进入红区的 ID
                    print(f"🟥 玩家 {track_id} 进入红色区域")
                elif not in_red and track_id in red_area_ids:
                    red_area_ids.remove(track_id)  # 离开红区
                    print(f"🚨 玩家 {track_id} 离开红色区域（判定闯红灯）")
                    if foot_speed > 10:
                        violation_log.add(track_id)
                        send_violation_count(len(violation_log))
                        violation_records.append({
                            'track_id': track_id,
                            'x': foot_x,
                            'y': foot_y,
                            'speed': round(foot_speed, 2),
                            'conf' : conf
                        })
                        cv2.putText(frame, f"ID {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.rectangle(frame, (l, t), (l + box_width, t + box_height), (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"ID {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.rectangle(frame, (l, t), (l + box_width, t + box_height), (255, 255, 0), 2)
        # light_color = (0, 0, 255) if red_light else (0, 255, 0)
        # cv2.rectangle(frame, (RED_LIGHT_ROI[0], RED_LIGHT_ROI[1]), (RED_LIGHT_ROI[2], RED_LIGHT_ROI[3]), light_color, 2)
        # cv2.polylines(frame, [ZEBRA_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)

        # out.write(frame)
        cv2.imshow("RTMP Stream", frame)
        key = cv2.waitKey(1)
        if key == ord("q") or not cv2.getWindowProperty("RTMP Stream", cv2.WND_PROP_VISIBLE):
            print("🛑 收到退出信号，关闭线程和资源...")
            break
        
    global stop_flag
    stop_flag = True
    reader_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()
    # out.release()
    with open('violations1.csv', 'w', newline='') as f:
        fieldnames = ['track_id', 'x', 'y', 'speed', 'conf']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in violation_records:
            writer.writerow(row)

    # 写入总人数统计到单独文件
    with open('summary1.csv', 'w', newline='') as f:
        summary_writer = csv.writer(f)
        summary_writer.writerow(['total_tracked_people', 'violation_people', 'total_people'])
        summary_writer.writerow([len(cross_log), len(violation_log), len(track_history)])


if __name__ == '__main__':
    process_video()
