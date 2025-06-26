import random
import time

from client_sender import send_extend_signal

def simulate_red_light_violation(arduino_sender, red_duration=5):
    """
    模拟红灯期间穿越检测逻辑
    - xt: 红灯开始时检测区域内人数
    - x1: 初始在区域内、后来离开的人数
    - xe: 红灯期间进入区域人数
    - x2: xe中离开的人数
    计算：(x1 + x2) / (xt + xe) > 0.5，则触发延长红灯
    """
    xt0 = random.randint(0, 4)  # 初始在框中的人数
    xt = xt0  # 当前框中初始人的剩余人数
    x1 = 0    # 初始人在红灯期间离开的数量
    xe = 0    # 红灯期间进入框内的总人数
    x2 = 0    # xe中又离开的人数

    # 状态列表用于追踪：True 表示还在框内
    entered_people_flags = []

    print(f"[红灯开始] 初始框内人数 xt = {xt0}")

    for t in range(red_duration):
        # 进入框的人数（0~2）
        new_entries = random.randint(0, 2)
        xe += new_entries
        entered_people_flags.extend([True] * new_entries)

        # 初始人群中最多离开 min(xt, 2) 人
        exits_from_initial = random.randint(0, min(xt, 2))
        x1 += exits_from_initial
        xt -= exits_from_initial

        # 后进人中离开 min(仍在框内人数, 2)
        available_exit_from_entered = entered_people_flags.count(True)
        exits_from_entered = random.randint(0, min(available_exit_from_entered, 2))
        exited = 0
        for i in range(len(entered_people_flags)):
            if entered_people_flags[i] and exited < exits_from_entered:
                entered_people_flags[i] = False
                exited += 1
        x2 += exited

        print(f"[第{t+1}秒] 新进: {new_entries}，初始出: {exits_from_initial}，后进出: {exited}")

        # 可加 sleep 模拟真实时间
        time.sleep(1)

    # 比例判断
    denominator = xt0 + xe
    numerator = x1 + x2
    ratio = numerator / denominator if denominator > 0 else 0

    print(f"[红灯结束] x1={x1}, x2={x2}, xt0={xt0}, xe={xe}，比例={(ratio):.2f}")

    if ratio > 0.5:
        print("⚠️ 超过 0.8，发送延长红灯信号")
        arduino_sender()
    else:
        print("✅ 比例合理，无需延长红灯")

# 测试运行
simulate_red_light_violation(send_extend_signal, red_duration=6)
