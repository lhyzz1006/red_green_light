import requests
import os
import datetime
import cv2

# 图片保存目录
screenshot_dir = "chaos_screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Unity监听的接收接口地址（替换为实际Unity机器IP和端口）
unity_endpoint = "http://172.20.10.8:8080/receive-image/"

def save_person_crop(frame, l, t, box_width, box_height, real_id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(screenshot_dir, f"chaos_{real_id}_{timestamp}.jpg")
    person_crop = frame[t:t + box_height, l:l + box_width]
    cv2.imwrite(filename, person_crop)
    print(f"[📸 截图已保存] {filename}")
    return filename

def send_image_to_unity(image_path):
    print(f"[🚀 正在发送图片给 Unity] {image_path}, IP:{unity_endpoint}")
    try:
        with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
        headers = {"Content-Type": "application/octet-stream"}
        response = requests.post(unity_endpoint, data=img_bytes, headers=headers)

        if response.status_code == 200:
            print("图片已发送给 Unity")
        else:
            print(f" Unity响应失败: {response.status_code}")
    except Exception as e:
        print(f"图片发送失败: {e}")
