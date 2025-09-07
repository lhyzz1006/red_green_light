import threading
import queue
import requests
import time
from light_controller import extend_light, jump_to, reset_light

arduino_ip = "172.20.10.5"
violation_queue = queue.Queue()

def sender_worker():
    while True:
        count = violation_queue.get()
        try:
            url = f"http://{arduino_ip}/chaos"
            r = requests.get(url, timeout=2)
            print(f"Sent {count}, Arduino replied: {r.text}")
        except Exception as e:
            print(f"Request failed: {e}")
        violation_queue.task_done()

# 启动后台线程（只启动一次）
threading.Thread(target=sender_worker, daemon=True).start()

# 提供给主程序的接口
def send_violation_count(count):
    if(count) >= 6:
        violation_queue.put("chaos")


# 示例 Arduino 通信函数
def send_extend_signal():
     extend_light("red", 5)