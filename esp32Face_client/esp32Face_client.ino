#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// WiFi credentials
const char* ssid = "print3d_2";
const char* password = "INNOVATE3D";

// PCA9685 setup
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo configuration
const int SERVO_CHANNEL = 0;  // PCA9685 channel for servo
const int SERVO_MIN = 150;    // Minimum pulse width (adjust for your servo)
const int SERVO_MAX = 600;    // Maximum pulse width (adjust for your servo)

// Current servo position
float currentAngle = 90.0;    // Start at center position
float targetAngle = 90.0;
unsigned long lastMoveTime = 0;
const unsigned long MOVE_DELAY = 15; // Smooth movement delay in ms

// Web server
WebServer server(80);

// Function prototypes
void setupWiFi();
void setupPCA9685();
void handleSetServo();
void handleGetStatus();
void handleRoot();
void handleNotFound();
float mapAngleToServo(float angle);
void moveServoSmooth();
void setServoAngle(float angle);

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Servo Controller Starting...");
  
  // Initialize I2C for PCA9685
  Wire.begin();
  
  // Setup PCA9685
  setupPCA9685();
  
  // Setup WiFi
  setupWiFi();
  
  // Setup web server routes
  server.on("/", HTTP_GET, handleRoot);
  server.on("/servo", HTTP_POST, handleSetServo);
  server.on("/status", HTTP_GET, handleGetStatus);
  server.onNotFound(handleNotFound);
  
  // Enable CORS for all responses
  server.enableCORS(true);
  
  // Start server
  server.begin();
  Serial.println("HTTP server started");
  Serial.print("Server IP: ");
  Serial.println(WiFi.localIP());
  
  // Set servo to center position
  setServoAngle(90.0);
  Serial.println("Servo initialized to center position (90 degrees)");
}

void loop() {
  server.handleClient();
  moveServoSmooth();
  delay(1);
}

void setupWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setupPCA9685() {
  pwm.begin();
  pwm.setPWMFreq(50);  // 50Hz for servos
  Serial.println("PCA9685 initialized");
  
  // Wait for PCA9685 to stabilize
  delay(100);
}

void handleRoot() {
  String html = "<!DOCTYPE html>";
  html += "<html>";
  html += "<head>";
  html += "<title>ESP32 Servo Controller</title>";
  html += "<style>";
  html += "body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }";
  html += ".container { max-width: 400px; margin: 0 auto; }";
  html += "input[type=range] { width: 100%; margin: 20px 0; }";
  html += "button { padding: 10px 20px; margin: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }";
  html += "button:hover { background: #45a049; }";
  html += ".angle-display { font-size: 24px; margin: 20px 0; }";
  html += "</style>";
  html += "</head>";
  html += "<body>";
  html += "<div class=\"container\">";
  html += "<h1>ESP32 Servo Controller</h1>";
  html += "<div class=\"angle-display\">";
  html += "Current Angle: <span id=\"currentAngle\">90</span>&deg;";
  html += "</div>";
  html += "<input type=\"range\" id=\"angleSlider\" min=\"0\" max=\"180\" value=\"90\" oninput=\"updateDisplay()\">";
  html += "<br>";
  html += "<button onclick=\"setAngle()\">Set Angle</button>";
  html += "<button onclick=\"centerServo()\">Center (90&deg;)</button>";
  html += "<br><br>";
  html += "<div id=\"status\">Ready</div>";
  html += "</div>";
  html += "<script>";
  html += "function updateDisplay() {";
  html += "const angle = document.getElementById('angleSlider').value;";
  html += "document.getElementById('currentAngle').textContent = angle;";
  html += "}";
  html += "function setAngle() {";
  html += "const angle = document.getElementById('angleSlider').value;";
  html += "fetch('/servo', {";
  html += "method: 'POST',";
  html += "headers: {'Content-Type': 'application/json'},";
  html += "body: JSON.stringify({angle: parseInt(angle)})";
  html += "})";
  html += ".then(response => response.json())";
  html += ".then(data => {";
  html += "document.getElementById('status').textContent = data.message || 'Angle set successfully';";
  html += "})";
  html += ".catch(error => {";
  html += "document.getElementById('status').textContent = 'Error: ' + error.message;";
  html += "});";
  html += "}";
  html += "function centerServo() {";
  html += "document.getElementById('angleSlider').value = 90;";
  html += "updateDisplay();";
  html += "setAngle();";
  html += "}";
  html += "setInterval(() => {";
  html += "fetch('/status')";
  html += ".then(response => response.json())";
  html += ".then(data => {";
  html += "if (data.current_angle !== undefined) {";
  html += "document.getElementById('currentAngle').textContent = data.current_angle.toFixed(1);";
  html += "}";
  html += "})";
  html += ".catch(error => console.log('Status update error:', error));";
  html += "}, 1000);";
  html += "</script>";
  html += "</body>";
  html += "</html>";
  
  server.send(200, "text/html", html);
}

void handleSetServo() {
  if (server.hasArg("plain")) {
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, server.arg("plain"));
    
    if (error) {
      server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
      return;
    }
    
    if (doc.containsKey("angle")) {
      float angle = doc["angle"];
      
      // Validate angle range
      if (angle < 0 || angle > 180) {
        server.send(400, "application/json", "{\"error\":\"Angle must be between 0 and 180\"}");
        return;
      }
      
      targetAngle = angle;
      Serial.printf("Target angle set to: %.1f degrees\n", targetAngle);
      
      DynamicJsonDocument response(200);
      response["status"] = "success";
      response["target_angle"] = targetAngle;
      response["current_angle"] = currentAngle;
      response["message"] = "Servo angle updated";
      
      String responseString;
      serializeJson(response, responseString);
      server.send(200, "application/json", responseString);
    } else {
      server.send(400, "application/json", "{\"error\":\"Missing angle parameter\"}");
    }
  } else {
    server.send(400, "application/json", "{\"error\":\"No data received\"}");
  }
}

void handleGetStatus() {
  DynamicJsonDocument doc(300);
  doc["status"] = "online";
  doc["current_angle"] = currentAngle;
  doc["target_angle"] = targetAngle;
  doc["wifi_rssi"] = WiFi.RSSI();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["uptime"] = millis();
  
  String response;
  serializeJson(doc, response);
  server.send(200, "application/json", response);
}

void handleNotFound() {
  String message = "File Not Found\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  message += "\nArguments: ";
  message += server.args();
  message += "\n";
  
  for (uint8_t i = 0; i < server.args(); i++) {
    message += " " + server.argName(i) + ": " + server.arg(i) + "\n";
  }
  
  server.send(404, "text/plain", message);
}

float mapAngleToServo(float angle) {
  // Map 0-180 degrees to servo pulse width
  // Adjust SERVO_MIN and SERVO_MAX values for your specific servo
  return map(angle * 10, 0, 1800, SERVO_MIN, SERVO_MAX);
}

void moveServoSmooth() {
  if (millis() - lastMoveTime > MOVE_DELAY) {
    if (abs(currentAngle - targetAngle) > 0.5) {
      // Move towards target angle
      float step = (targetAngle - currentAngle) * 0.1; // Smooth movement factor
      
      // Minimum step size to avoid getting stuck
      if (abs(step) < 0.5) {
        step = (targetAngle > currentAngle) ? 0.5 : -0.5;
      }
      
      currentAngle += step;
      
      // Ensure we don't overshoot
      if ((step > 0 && currentAngle > targetAngle) || 
          (step < 0 && currentAngle < targetAngle)) {
        currentAngle = targetAngle;
      }
      
      // Set the servo position
      setServoAngle(currentAngle);
      lastMoveTime = millis();
    }
  }
}

void setServoAngle(float angle) {
  // Constrain angle to valid range
  angle = constrain(angle, 0, 180);
  
  // Convert angle to pulse width
  uint16_t pulseWidth = mapAngleToServo(angle);
  
  // Set PWM on the PCA9685
  pwm.setPWM(SERVO_CHANNEL, 0, pulseWidth);
  
  // Debug output
  if (abs(angle - currentAngle) > 0.1) {
    Serial.printf("Servo angle: %.1f degrees (pulse: %d)\n", angle, pulseWidth);
  }
}