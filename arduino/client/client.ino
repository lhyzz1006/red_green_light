#include <WiFiS3.h>

const char* ssid = "Maocong";
const char* password = "lhyzzzzjx";

WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // 连接 WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    String req = client.readStringUntil('\r');
    client.flush();

    if (req.indexOf("GET /update?num=") != -1) {
      int pos = req.indexOf("num=") + 4;
      int val = req.substring(pos).toInt();
      Serial.print("counts：");
      Serial.println(val);
      // 在这里处理 LED 等操作

      client.println("HTTP/1.1 200 OK");
      client.println("Content-Type: text/plain");
      client.println();
      client.println("OK");
    }
    client.stop();
  }
}
