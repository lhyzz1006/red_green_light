import requests

ARDUINO_IP = "http://192.168.1.183" 

def extend_light(color="red", seconds=2):
    """
    向 Arduino 发送延长红/绿灯的请求
    :param color: "red" or "green"
    :param seconds: 延长时间，单位秒
    """
    try:
        url = f"{ARDUINO_IP}/?color={color}&time={seconds}"
        response = requests.get(url, timeout=2)
        print("响应：", response.text.strip())
    except requests.exceptions.RequestException as e:
        print("通信失败：", e)

def jump_to(target="red"):
    """
    强制跳转灯光状态（测试用）
    :param target: "red" or "green"
    """
    try:
        url = f"{ARDUINO_IP}/jump?target={target}"
        response = requests.get(url, timeout=2)
        print("跳转响应：", response.text.strip())
    except requests.exceptions.RequestException as e:
        print("跳转失败：", e)

def reset_light():
    """
    重置为红灯，清除所有延时
    """
    try:
        url = f"{ARDUINO_IP}/reset"
        response = requests.get(url, timeout=2)
        print("重置响应：", response.text.strip())
    except requests.exceptions.RequestException as e:
        print("重置失败：", e)
