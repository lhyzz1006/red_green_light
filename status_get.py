import requests
def get_arduino_status():
    try:
        res = requests.get("http://172.20.10.5/status", timeout=0.5)  # 根据实际 IP
        if res.status_code == 200:
            return res.json().get("color", "")
    except Exception as e:
        print("⚠️ 无法获取 Arduino 状态:", e)
    return ""