import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import multiprocessing as mp
import threading
import queue
import time

from client_sender import send_violation_count
from screenshot_sender import save_person_crop, send_image_to_unity
from reid_manager import ReIDManager
# from status_get import get_arduino_status

CONFIDENCE_THRESHOLD = 0.5
tile_model = YOLO('F:\\signal\\code\\runs\\detect\\train5\\weights\\best.pt')

from unity_http_rc import start_flask_server, register_callback
import threading

shared_yolo = YOLO('yolov8n.pt')
# 用于主程序实时更新状态的变量
manual_count_from_unity = 0

def on_unity_count_updated(count):
    global manual_count_from_unity
    manual_count_from_unity = count
    print(f"📥 RTMP主程序收到Unity推送人数: {count}")
    # ✅ 你也可以在这里直接触发 chaos 检查等逻辑
    if count > 9:
        print("⚠️ 超过混沌阈值，发送 chaos 请求...")
        import requests
        try:
            requests.get("http://172.20.10.5/chaos", timeout=1)
        except Exception as e:
            print("❌ chaos 请求失败：", e)

# 启动 Flask + 注册回调
def launch_flask():
    register_callback(on_unity_count_updated)
    threading.Thread(target=start_flask_server, daemon=True).start()

def detect_map_tiles(frame, conf=0.5):
    results = tile_model.predict(frame, conf=conf)[0]
    tile_result = []
    for box in results.boxes:
        cls = int(box.cls[0])
        name = tile_model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tile_result.append({'name': name, 'bbox': (x1, y1, x2, y2)})
    return tile_result

def freeze_tile_regions(tile_results, frozen_regions, current_time):
    for tile in tile_results:
        x1, y1, x2, y2 = tile['bbox']
        label = tile['name']
        if label in frozen_regions and frozen_regions[label]['bbox'] is None:
            frozen_regions[label]['bbox'] = (x1, y1, x2, y2)
            frozen_regions[label]['timestamp'] = current_time

def draw_frozen_regions(frame, frozen_regions):
    for name, info in frozen_regions.items():
        if info['bbox']:
            x1, y1, x2, y2 = info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

def process_video(video_path, stream_id):
    yolo_model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=5)
    reid_manager = ReIDManager(threshold=0.5)
    region_presence = {}
    chaos_log = set()
    frozen_regions = {
        'oxford': {'bbox': None, 'timestamp': None},
        'park': {'bbox': None, 'timestamp': None}
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频流: {video_path}")
        return

    buffer = queue.Queue(maxsize=5)

    def reader():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            if buffer.full():
                try:
                    buffer.get_nowait()
                except queue.Empty:
                    pass
            buffer.put(frame)

    threading.Thread(target=reader, daemon=True).start()

    window_name = f"Stream {stream_id}"
    while True:
        try:
            frame = buffer.get(timeout=1)
        except queue.Empty:
            continue

        current_time = time.time()
        tile_results = detect_map_tiles(frame, conf=0.5)
        freeze_tile_regions(tile_results, frozen_regions, current_time)
        draw_frozen_regions(frame, frozen_regions)

        results = yolo_model(frame)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                if 1.0 * (y2 - y1) / (x2 - x1) < 1.5:
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            box_width = max(1, w - l)
            box_height = max(1, h - t)

            foot_x = int(l + box_width / 2)
            foot_y = int(t + box_height * 0.95)
            img_h, img_w = frame.shape[:2]
            crop_t, crop_b = max(0, t), min(foot_y, img_h)
            crop_l, crop_r = max(0, l), min(w, img_w)
            if crop_b <= crop_t or crop_r <= crop_l:
                continue

            crop_img = frame[crop_t:crop_b, crop_l:crop_r]
            if crop_img is None or crop_img.size == 0:
                continue

            reid_manager.update_feature(track_id, crop_img)
            if track_id not in region_presence:
                region_presence[track_id] = False

            in_frozen_region = any(
                x1 <= foot_x <= x2 and y1 <= foot_y <= y2
                for reg in frozen_regions.values() if reg['bbox']
                for x1, y1, x2, y2 in [reg['bbox']]
            )
            
            # 获取灯状态，如果是 chaos 就跳过该帧
            # light_state = get_arduino_status()
            # if light_state == "chaos":
            #     cv2.putText(frame, "CHAOS MODE - SKIP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            #     cv2.imshow(window_name, frame)
            #     if cv2.waitKey(1) == ord('q'):
            #         break
            #     continue

            if region_presence.get(track_id, False) and not in_frozen_region:
                if track_id not in chaos_log:
                    chaos_log.add(track_id)
                    image_path = save_person_crop(frame, l, t, box_width, box_height, track_id)
                    send_image_to_unity(image_path)
                region_presence[track_id] = False
                cv2.rectangle(frame, (l, t), (l + box_width, t + box_height), (0, 0, 255), 2)
            elif not region_presence.get(track_id, False) and in_frozen_region:
                region_presence[track_id] = True
                cv2.rectangle(frame, (l, t), (l + box_width, t + box_height), (0, 255, 0), 2)
            else:
                color = (0, 255, 0) if in_frozen_region else (255, 255, 0)
                cv2.rectangle(frame, (l, t), (l + box_width, t + box_height), color, 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_multiple_streams(video_paths):
    mp.set_start_method('spawn', force=True)
    processes = []
    for i, path in enumerate(video_paths):
        p = mp.Process(target=process_video, args=(path, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == '__main__':
    launch_flask()  # 启动 Flask 并注册回调 
    time.sleep(5)
    video_paths = [
        'rtmp://127.0.0.1/live/stream',
        'rtmp://127.0.0.1/live/stream1'
    ]
    process_multiple_streams(video_paths)

