# client_sender.py
import threading
import queue
import requests
import time

arduino_ip = "172.20.10.13"  # 替换为你的 Arduino IP
violation_queue = queue.Queue()

def sender_worker():
    while True:
        count = violation_queue.get()
        try:
            url = f"http://{arduino_ip}/update?num={count}"
            r = requests.get(url, timeout=2)
            print(f"✅ Sent {count}, Arduino replied: {r.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")
        violation_queue.task_done()

# 启动后台线程（只启动一次）
threading.Thread(target=sender_worker, daemon=True).start()

# 提供给主程序的接口
def send_violation_count(count):
    violation_queue.put(count)
