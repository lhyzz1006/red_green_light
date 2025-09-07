#include <WiFiS3.h>

char ssid[] = "Maocong";
char pass[] = "lhyzzzzjx";

WiFiServer server(80);

// 灯引脚
const int redPin = 2;
const int greenPin = 3;

// 时间设置
int redBaseDuration = 3000;
int greenBaseDuration = 5000;
int redAddedTime = 0;
int greenAddedTime = 0;
bool isChaos = false;

unsigned long lastChangeTime = 0;
unsigned long lastPrintTime = 0;
const int printInterval = 1000;

enum LightState { RED,
                  GREEN,
                  CHAOS };
LightState currentState = RED;

IPAddress local_ip(172, 20, 10, 5);
IPAddress gateway(172, 20, 10, 1);
IPAddress subnet(255, 255, 255, 240);

void setup() {
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);

  Serial.begin(9600);
  WiFi.config(local_ip, gateway, subnet);
  WiFi.begin(ssid, pass);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.print("\nIP地址: ");
  Serial.println(WiFi.localIP());

  server.begin();
  lastChangeTime = millis();
  updateLights();
}

void loop() {
  handleClient();  // 非阻塞处理 HTTP 请求

  unsigned long now = millis();
  int totalDuration = (currentState == RED)
                        ? redBaseDuration + redAddedTime
                        : greenBaseDuration + greenAddedTime;

  if (currentState != CHAOS && now - lastChangeTime >= totalDuration) {
    switchLight();
    lastChangeTime = now;
    updateLights();
  }

  if (now - lastPrintTime >= printInterval) {
    lastPrintTime = now;
    int remaining = (totalDuration - (now - lastChangeTime)) / 1000;
    if (remaining < 0) remaining = 0;

    Serial.print("状态: ");
    Serial.print((currentState == RED) ? "RED" : "GREEN");
    Serial.print(" 剩余: ");
    Serial.print(remaining);
    Serial.println(" 秒");
  }
}

void updateLights() {
  digitalWrite(redPin, currentState == RED ? HIGH : LOW);
  digitalWrite(greenPin, currentState == GREEN ? HIGH : LOW);
}

void switchLight() {
  currentState = (currentState == RED) ? GREEN : RED;
}

void sendHttpResponse(WiFiClient& client, const String& msg, bool isJson = false) {
  client.println("HTTP/1.1 200 OK");
  client.print("Content-Type: ");
  client.println(isJson ? "application/json" : "text/plain");
  client.println("Connection: close");
  client.println();
  client.println(msg);
  delay(2);  // 小延迟避免数据未发完即关闭
  client.stop();
}

void handleClient() {
  WiFiClient client = server.available();
  if (!client) return;

  unsigned long timeout = millis();
  String request = "";

  while (client.connected() && millis() - timeout < 1000) {
    if (client.available()) {
      char c = client.read();
      request += c;
      if (c == '\n') break;
    }
  }

  request.trim();
  request.toLowerCase();


  if (request.indexOf("get /status") >= 0) {

    String color;
    if (currentState == CHAOS) color = "chaos";
    else color = (currentState == RED) ? "red" : "green";
    int totalDuration = (currentState == RED)
                          ? redBaseDuration + redAddedTime
                          : greenBaseDuration + greenAddedTime;
    int remaining = (currentState == CHAOS) ? -1 : (totalDuration - (millis() - lastChangeTime)) / 1000;
    if (remaining < 0) remaining = 0;

    String json = "{\"color\":\"" + color + "\",\"remaining\":" + String(remaining) + "}";
    sendHttpResponse(client, json);
    return;
  }

  if (request.indexOf("get /jump") >= 0) {
    String target = getValue(request, "target");
    if (target == "red") currentState = RED;
    else if (target == "green") currentState = GREEN;

    lastChangeTime = millis();
    updateLights();
    sendHttpResponse(client, "跳转成功");
    return;
  }

  if (request.indexOf("get /reset") >= 0) {
    currentState = RED;
    redAddedTime = 0;
    greenAddedTime = 0;
    lastChangeTime = millis();
    updateLights();
    String response = "{\"status\":\"reset\",\"confirmed\":true}";
    sendHttpResponse(client, response);
    return;
  }

  if (request.indexOf("get /chaos") >= 0) {
    currentState = CHAOS;
    isChaos = true;
    updateLights();

    String source = getValue(request, "source");  // 获取来源

    // 返回给触发者的确认信息
    String response = "{\"status\":\"chaos\",\"confirmed\":true,\"from\":\"" + source + "\"}";
    sendHttpResponse(client, response);
    return;
  }
  if (request.indexOf("get /exitchaos") >= 0) {
    currentState = RED;
    isChaos = false;
    lastChangeTime = millis();
    updateLights();

    String response = "{\"status\":\"exitChaos\",\"confirmed\":true}";
    sendHttpResponse(client, response, true);
    return;
  }


  String color = getValue(request, "color");
  int timeExtra = getValue(request, "time").toInt();

  if (color == "red") {
    redAddedTime += timeExtra * 1000;
    sendHttpResponse(client, "红灯延长成功");
  } else if (color == "green") {
    greenAddedTime += timeExtra * 1000;
    sendHttpResponse(client, "绿灯延长成功");
  } else {
    sendHttpResponse(client, "无效请求");
  }
}


String getValue(String data, String key) {
  int start = data.indexOf(key + "=");
  if (start == -1) return "";
  start += key.length() + 1;
  int end = data.indexOf('&', start);
  if (end == -1) end = data.indexOf(' ', start);
  if (end == -1) end = data.length();
  return data.substring(start, end);
}