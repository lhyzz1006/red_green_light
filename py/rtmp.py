import asyncio
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import multiprocessing as mp
import threading
import queue
import time
import requests

from client_sender import send_violation_count
from screenshot_sender import save_person_crop, send_image_to_unity
from reid_manager import ReIDManager
from unity_http_rc import start_flask_server, register_callback

CONFIDENCE_THRESHOLD = 0.5
tile_model = YOLO('F:\\signal\\code\\runs\\detect\\train5\\weights\\best.pt')
# tile_model = YOLO('F:\\signal\\code\\model\\best.pt')
shared_yolo = YOLO('yolov8n.pt')

manual_count_from_unity = 0

# def on_unity_count_updated(count):
#     global manual_count_from_unity
#     manual_count_from_unity = count
#     print(f"\U0001f4e5 RTMP主程序收到Unity推送人数: {count}")
#     if count > 9:
#         print("超过混沌阈值，发送 chaos 请求...")
#         try:
#             requests.get("http://172.20.10.5/chaos", timeout=1)
#         except Exception as e:
#             print("chaos 请求失败：", e)

# def launch_flask():
#     register_callback(on_unity_count_updated)
#     threading.Thread(target=start_flask_server, daemon=True).start()

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
    frozen_regions = {
        'oxford': {'bbox': None, 'timestamp': None},
        'park': {'bbox': None, 'timestamp': None}
    }

    person_state = {}
    last_status_check_time = 0
    cached_light_state = "unknown"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频流: {video_path}")
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
        tile_results = detect_map_tiles(frame, conf=0.6)
        if tile_results is not None:
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
                if 1.0 * (y2 - y1) / (x2 - x1) < 1:
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        if current_time - last_status_check_time >= 2:
            try:
                response = requests.get("http://172.20.10.5/status", timeout=0.5)
                data = response.json()  # 解析 JSON
                cached_light_state = data.get("color", "unknown")
            except Exception as e:
                print("灯状态请求异常：", e)
                cached_light_state = "unknown"
            last_status_check_time = current_time

        # cached_light_state = "red"
        if  cached_light_state != "red":
            cv2.putText(frame, f"Light: {cached_light_state.upper()} - SKIP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue
        

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = map(int, track.to_ltrb())
            box_width = max(1, w - l)
            box_height = max(1, h - t)

            foot_x = int(l + box_width / 2)
            foot_y = int(t + box_height * 0.95)

            crop_img = frame[max(0, t):min(foot_y, frame.shape[0]), max(0, l):min(w, frame.shape[1])]
            if crop_img is not None and crop_img.size > 0:
                reid_manager.update_feature(track_id, crop_img)

            in_frozen_region = any(
                x1 <= foot_x <= x2 and y1 <= foot_y <= y2
                for reg in frozen_regions.values() if reg['bbox']
                for x1, y1, x2, y2 in [reg['bbox']]
            )

            if track_id not in person_state:
                person_state[track_id] = {'entered_time': None, 'in_region': False}

            state = person_state[track_id]

            if in_frozen_region:
                if not state['in_region']:
                    state['entered_time'] = current_time
                    state['in_region'] = True
                else:
                    duration = current_time - state['entered_time']
                    if duration >= 5:
                        image_path = save_person_crop(frame, l, t, box_width, box_height, track_id)
                        send_image_to_unity(image_path)
                        state['entered_time'] = current_time
            else:
                if state['in_region']:
                    duration = current_time - state['entered_time']
                    if duration >= 2:
                        image_path = save_person_crop(frame, l, t, box_width, box_height, track_id)
                        send_image_to_unity(image_path)
                    state['in_region'] = False
                    state['entered_time'] = None

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
    # launch_flask()
    time.sleep(5)
    video_paths = [
        'rtmp://127.0.0.1/live/stream',
        'rtmp://127.0.0.1/live/stream1'
    ]
    process_multiple_streams(video_paths)
