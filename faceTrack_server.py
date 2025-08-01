import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
from collections import deque
import time
import asyncio
import json
from typing import Optional
import requests
import threading
from queue import Queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

class ServoController:
    def __init__(self, esp32_ip: str = "192.168.1.88"):
        self.esp32_ip = esp32_ip
        self.current_angle = 90  # Default center position
        self.target_angle = 90
        self.min_angle = 0
        self.max_angle = 180
        self.command_queue = Queue()
        self.is_connected = False
        
    def test_connection(self) -> bool:
        """Test if ESP32 is reachable"""
        try:
            response = requests.get(f"http://{self.esp32_ip}/status", timeout=2)
            self.is_connected = response.status_code == 200
            return self.is_connected
        except:
            self.is_connected = False
            return False
    
    def set_angle(self, angle: float):
        """Set servo angle with bounds checking"""
        angle = max(self.min_angle, min(self.max_angle, angle))
        self.target_angle = angle
        
        try:
            payload = {"angle": angle}
            response = requests.post(
                f"http://{self.esp32_ip}/servo", 
                json=payload, 
                timeout=1
            )
            if response.status_code == 200:
                self.current_angle = angle
                return True
        except Exception as e:
            print(f"Servo control error: {e}")
        return False
    
    def center_servo(self):
        """Move servo to center position"""
        return self.set_angle(90)

class FaceTrackingServo:
    def __init__(self, esp32_ip: str = "192.168.1.88"):
        self.model_path = 'face_landmarker_v2_with_blendshapes.task'
        self.download_model()
        self.init_face_landmarker()
        
        # Tracking parameters
        self.landmark_history = deque(maxlen=5)
        self.fps_history = deque(maxlen=10)  # Reduced for faster calculation
        
        # Servo control
        self.servo = ServoController(esp32_ip)
        
        # Face tracking parameters
        self.frame_center_x = 240  # Updated for 480px width
        self.tracking_enabled = True
        self.face_center_history = deque(maxlen=5)  # Reduced history for faster response
        self.face_detected_frames = 0  # Track consecutive face detections
        self.min_detection_frames = 5  # Reduced from 10 for faster response
        
        # PID-like control parameters
        self.kp = 0.3  # Proportional gain
        self.deadzone = 40  # Pixels - adjusted for smaller resolution
        self.max_movement = 10  # Max degrees to move per frame
        
    def download_model(self):
        if not os.path.exists(self.model_path):
            print("Downloading face landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_with_blendshapes/float16/latest/face_landmarker_v2_with_blendshapes.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Model downloaded successfully!")

    def init_face_landmarker(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def smooth_landmarks(self, current_landmarks):
        self.landmark_history.append(current_landmarks)
        num_frames = len(self.landmark_history)
        smoothed = []
        for i in range(len(current_landmarks)):
            x = sum(history[i].x for history in self.landmark_history) / num_frames
            y = sum(history[i].y for history in self.landmark_history) / num_frames
            z = sum(history[i].z for history in self.landmark_history) / num_frames
            landmark = type(current_landmarks[i])()
            landmark.x, landmark.y, landmark.z = x, y, z
            smoothed.append(landmark)
        return smoothed

    def get_face_center(self, face_landmarks, width, height):
        """Calculate the center point of the detected face"""
        if not face_landmarks:
            return None
            
        # Use key facial points to determine face center
        key_points = [1, 9, 10, 151, 175, 176]  # Nose tip, chin, forehead points
        
        x_coords = [face_landmarks[i].x * width for i in key_points if i < len(face_landmarks)]
        y_coords = [face_landmarks[i].y * height for i in key_points if i < len(face_landmarks)]
        
        if x_coords and y_coords:
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            return (center_x, center_y)
        return None

    def calculate_servo_adjustment(self, face_center_x, frame_width):
        """Calculate how much to adjust the servo based on face position"""
        if not self.tracking_enabled:
            return 0
            
        # Calculate error from center
        center_x = frame_width / 2
        error = face_center_x - center_x
        
        # Apply deadzone
        if abs(error) < self.deadzone:
            return 0
            
        # Calculate proportional response
        # Positive error means face is to the right, so servo should move right (increase angle)
        adjustment = self.kp * (error / center_x) * 45  # Scale to reasonable servo range
        
        # Limit maximum movement per frame
        adjustment = max(-self.max_movement, min(self.max_movement, adjustment))
        
        return adjustment

    def update_servo_position(self, face_center_x, frame_width):
        """Update servo position to track face"""
        if face_center_x is None:
            # Clear history when no face detected
            self.face_center_history.clear()
            return
            
        # Smooth the face center position
        self.face_center_history.append(face_center_x)
        if len(self.face_center_history) > 2:  # Reduced requirement for faster response
            smoothed_center = sum(self.face_center_history) / len(self.face_center_history)
        else:
            # Don't move servo until we have minimal history
            return
            
        # Calculate servo adjustment
        adjustment = self.calculate_servo_adjustment(smoothed_center, frame_width)
        
        # Only move if adjustment is significant AND we have minimal tracking
        if abs(adjustment) > 0.5 and len(self.face_center_history) >= 3:  # Reduced from 5
            new_angle = self.servo.current_angle + adjustment
            self.servo.set_angle(new_angle)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        if not detection_result.face_landmarks:
            # Reset face detection counter when no face found
            self.face_detected_frames = 0
            self.face_center_history.clear()
            return rgb_image

        annotated_image = np.copy(rgb_image)
        height, width = annotated_image.shape[:2]
        self.frame_center_x = width / 2

        # Increment face detection counter
        self.face_detected_frames += 1

        for face_landmarks in detection_result.face_landmarks:
            smoothed_landmarks = self.smooth_landmarks(face_landmarks)
            
            # Get face center for servo control
            face_center = self.get_face_center(smoothed_landmarks, width, height)
            if face_center:
                face_center_x, face_center_y = face_center
                
                # Draw face center point
                center_color = (255, 0, 255) if self.face_detected_frames < self.min_detection_frames else (0, 255, 0)
                cv2.circle(annotated_image, 
                          (int(face_center_x), int(face_center_y)), 
                          5, center_color, -1)
                
                # Draw center line
                cv2.line(annotated_image, 
                        (int(width/2), 0), 
                        (int(width/2), height), 
                        (255, 255, 255), 1)
                
                # Draw deadzone area
                deadzone_left = int(width/2 - self.deadzone)
                deadzone_right = int(width/2 + self.deadzone)
                cv2.rectangle(annotated_image, 
                            (deadzone_left, 10), 
                            (deadzone_right, 30), 
                            (0, 255, 255), 2)
                cv2.putText(annotated_image, 'DEADZONE', 
                           (deadzone_left, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Only update servo position if we have stable detection
                if self.face_detected_frames >= self.min_detection_frames:
                    self.update_servo_position(face_center_x, width)

            def draw_polyline(index_list, color, thickness):
                for i in range(len(index_list) - 1):
                    pt1 = (int(smoothed_landmarks[index_list[i]].x * width),
                           int(smoothed_landmarks[index_list[i]].y * height))
                    pt2 = (int(smoothed_landmarks[index_list[i+1]].x * width),
                           int(smoothed_landmarks[index_list[i+1]].y * height))
                    cv2.line(annotated_image, pt1, pt2, color, thickness)

            # Draw facial landmarks
            face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                         365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                         132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
            left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                        159, 160, 161, 246, 33]
            right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
                         387, 386, 385, 384, 398, 362]
            lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]

            draw_polyline(face_oval, (0, 255, 0), 1)
            draw_polyline(left_eye, (255, 0, 0), 1)
            draw_polyline(right_eye, (255, 0, 0), 1)
            draw_polyline(lips, (0, 0, 255), 2)

        return annotated_image

    def calculate_fps(self, start_time):
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        return np.mean(self.fps_history)

    def draw_info_overlay(self, image, fps, detection_result):
        height, width = image.shape[:2]
        
        # Status indicators
        cv2.putText(image, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        face_count = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
        cv2.putText(image, f'Faces: {face_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Servo status
        servo_color = (0, 255, 0) if self.servo.is_connected else (0, 0, 255)
        cv2.putText(image, f'Servo: {self.servo.current_angle:.1f}°', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, servo_color, 2)
        
        tracking_status = "ON" if self.tracking_enabled else "OFF"
        tracking_color = (0, 255, 0) if self.tracking_enabled else (0, 255, 255)
        cv2.putText(image, f'Tracking: {tracking_status}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
        
        # Face detection stability indicator
        stability_text = f'Stability: {self.face_detected_frames}/{self.min_detection_frames}'
        stability_color = (0, 255, 0) if self.face_detected_frames >= self.min_detection_frames else (0, 255, 255)
        cv2.putText(image, stability_text, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1)
        
        # Connection status
        conn_status = "Connected" if self.servo.is_connected else "Disconnected"
        conn_color = (0, 255, 0) if self.servo.is_connected else (0, 0, 255)
        cv2.putText(image, f'ESP32: {conn_status}', (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1)

# FastAPI application
app = FastAPI(title="Face Tracking Servo Controller")

# Global tracker instance
tracker: Optional[FaceTrackingServo] = None
camera = None

@app.on_event("startup")
async def startup_event():
    global tracker
    # Initialize with default ESP32 IP, can be changed via API
    tracker = FaceTrackingServo("192.168.1.100")
    print("Face tracking servo controller initialized")

@app.get("/")
async def get_dashboard():
    """Serve the dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Tracking Servo Controller</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .controls { display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }
            button { padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            .btn-warning { background: #ffc107; color: black; }
            .btn-danger { background: #dc3545; color: white; }
            .status { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }
            input { padding: 5px; margin: 0 5px; }
            #video { width: 100%; max-width: 640px; border: 2px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Face Tracking Servo Controller</h1>
            
            <div class="status">
                <strong>Status:</strong> 
                <span id="status">Initializing...</span>
            </div>
            
            <div class="controls">
                <button class="btn-primary" onclick="startTracking()">Start Camera</button>
                <button class="btn-danger" onclick="stopTracking()">Stop Camera</button>
                <button class="btn-success" onclick="toggleTracking()">Toggle Tracking</button>
                <button class="btn-warning" onclick="centerServo()">Center Servo</button>
            </div>
            
            <div class="controls">
                <label>ESP32 IP:</label>
                <input type="text" id="esp32_ip" value="192.168.1.100" placeholder="192.168.1.100">
                <button class="btn-primary" onclick="updateIP()">Update IP</button>
                <button class="btn-success" onclick="testConnection()">Test Connection</button>
            </div>
            
            <div class="controls">
                <label>Manual Servo Control:</label>
                <input type="range" id="servo_angle" min="0" max="180" value="90" oninput="updateServoDisplay()">
                <span id="servo_value">90°</span>
                <button class="btn-primary" onclick="setServoAngle()">Set Angle</button>
            </div>
            
            <img id="video" src="/video_feed" alt="Video stream will appear here when started">
            
            <div style="margin-top: 20px;">
                <h3>Controls:</h3>
                <ul>
                    <li><strong>Start Camera:</strong> Begin video streaming and face detection</li>
                    <li><strong>Toggle Tracking:</strong> Enable/disable automatic servo tracking</li>
                    <li><strong>Center Servo:</strong> Move servo to 90° position</li>
                    <li><strong>Manual Control:</strong> Override automatic tracking</li>
                </ul>
            </div>
        </div>

        <script>
            function startTracking() {
                fetch('/start_camera', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => updateStatus(data.message));
            }

            function stopTracking() {
                fetch('/stop_camera', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => updateStatus(data.message));
            }

            function toggleTracking() {
                fetch('/toggle_tracking', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => updateStatus(data.message));
            }

            function centerServo() {
                fetch('/center_servo', {method: 'POST'})
                    .then(r => r.json())
                    .then(data => updateStatus(data.message));
            }

            function updateIP() {
                const ip = document.getElementById('esp32_ip').value;
                fetch('/set_esp32_ip', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ip: ip})
                })
                .then(r => r.json())
                .then(data => updateStatus(data.message));
            }

            function testConnection() {
                fetch('/test_connection', {method: 'GET'})
                    .then(r => r.json())
                    .then(data => updateStatus(data.message));
            }

            function updateServoDisplay() {
                const angle = document.getElementById('servo_angle').value;
                document.getElementById('servo_value').textContent = angle + '°';
            }

            function setServoAngle() {
                const angle = document.getElementById('servo_angle').value;
                fetch('/set_servo_angle', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({angle: parseInt(angle)})
                })
                .then(r => r.json())
                .then(data => updateStatus(data.message));
            }

            function updateStatus(message) {
                document.getElementById('status').textContent = message;
            }

            // Update status periodically
            setInterval(() => {
                fetch('/status')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('status').textContent = data.status;
                    });
            }, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/start_camera")
async def start_camera():
    global camera
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            # Optimized camera settings for performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            camera.set(cv2.CAP_PROP_FPS, 24)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize lag
            # Additional optimizations
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        return {"message": "Camera started with optimized settings (480x360@24fps)"}
    except Exception as e:
        return {"message": f"Error starting camera: {str(e)}"}

@app.post("/stop_camera")
async def stop_camera():
    global camera
    try:
        if camera:
            camera.release()
            camera = None
        return {"message": "Camera stopped"}
    except Exception as e:
        return {"message": f"Error stopping camera: {str(e)}"}

@app.post("/toggle_tracking")
async def toggle_tracking():
    if tracker:
        tracker.tracking_enabled = not tracker.tracking_enabled
        status = "enabled" if tracker.tracking_enabled else "disabled"
        return {"message": f"Tracking {status}"}
    return {"message": "Tracker not initialized"}

@app.post("/center_servo")
async def center_servo():
    if tracker and tracker.servo.center_servo():
        return {"message": "Servo centered to 90°"}
    return {"message": "Failed to center servo"}

@app.post("/set_esp32_ip")
async def set_esp32_ip(data: dict):
    if tracker and "ip" in data:
        tracker.servo.esp32_ip = data["ip"]
        connection_ok = tracker.servo.test_connection()
        status = "connected" if connection_ok else "connection failed"
        return {"message": f"ESP32 IP updated to {data['ip']} - {status}"}
    return {"message": "Invalid request"}

@app.post("/set_servo_angle")
async def set_servo_angle(data: dict):
    if tracker and "angle" in data:
        success = tracker.servo.set_angle(data["angle"])
        if success:
            return {"message": f"Servo angle set to {data['angle']}°"}
        else:
            return {"message": "Failed to set servo angle"}
    return {"message": "Invalid request"}

@app.get("/test_connection")
async def test_connection():
    if tracker:
        connected = tracker.servo.test_connection()
        status = "ESP32 connected successfully" if connected else "ESP32 connection failed"
        return {"message": status}
    return {"message": "Tracker not initialized"}

@app.get("/status")
async def get_status():
    if tracker:
        servo_status = "Connected" if tracker.servo.is_connected else "Disconnected"
        tracking_status = "ON" if tracker.tracking_enabled else "OFF"
        return {
            "status": f"Servo: {servo_status} | Tracking: {tracking_status} | Angle: {tracker.servo.current_angle:.1f}°"
        }
    return {"status": "Tracker not initialized"}

def generate_frames():
    global camera, tracker
    
    # Frame skip for performance
    frame_skip = 0
    process_every_n_frames = 2  # Process every 2nd frame for face detection
    last_detection_result = None
    
    while True:
        if camera is None:
            # Send a placeholder frame
            placeholder = np.zeros((360, 480, 3), dtype=np.uint8)  # Updated size
            cv2.putText(placeholder, 'Camera not started', (150, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)  # Reduced sleep time
            continue
            
        start_time = time.time()
        ret, frame = camera.read()
        
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        
        # Skip face detection on some frames for performance
        frame_skip += 1
        if frame_skip >= process_every_n_frames:
            frame_skip = 0
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                if tracker is not None and hasattr(tracker, "detector"):
                    detection_result = tracker.detector.detect(mp_image)
                    last_detection_result = detection_result
                else:
                    raise AttributeError("Tracker or detector not initialized")
            except Exception as e:
                print(f"Detection error: {e}")
                class DummyDetectionResult:
                    def __init__(self):
                        self.face_landmarks = []
                detection_result = DummyDetectionResult()
                last_detection_result = detection_result
        else:
            # Use last detection result for non-processed frames
            detection_result = last_detection_result if last_detection_result else type('obj', (object,), {'face_landmarks': []})()

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if tracker is not None:
                annotated_image = tracker.draw_landmarks_on_image(rgb_frame, detection_result)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            else:
                annotated_image = frame
        except:
            annotated_image = frame

        if tracker is not None:
            fps = tracker.calculate_fps(start_time)
            tracker.draw_info_overlay(annotated_image, fps, detection_result)
        else:
            fps = 0

        # Compress image more for faster streaming
        ret, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting Face Tracking Servo Controller...")
    print("Dashboard will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)