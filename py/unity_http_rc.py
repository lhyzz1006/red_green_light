from flask import Flask, request, jsonify

app = Flask(__name__)

manual_count = 0
on_count_updated = None  # 外部注册的回调函数

@app.route('/set_count', methods=['POST'])
def set_count():
    global manual_count, on_count_updated
    data = request.get_json()
    if not data or 'count' not in data:
        return jsonify({"error": "Missing 'count'"}), 400

    try:
        manual_count = int(data['count'])
        print(f"Unity 设置人数: {manual_count}")

        if on_count_updated:
            on_count_updated(manual_count)  # 立即推送给主程序

        return jsonify({"status": "ok", "received": manual_count}), 200
    except ValueError:
        return jsonify({"error": "Invalid count"}), 400

def start_flask_server():
    app.run(host="0.0.0.0", port=5010)

def register_callback(func):
    global on_count_updated
    on_count_updated = func
