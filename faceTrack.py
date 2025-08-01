import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
from collections import deque
import time

class LiveFaceTracker:
    def __init__(self):
        self.model_path = 'face_landmarker_v2_with_blendshapes.task'
        self.download_model()
        self.init_face_landmarker()
        self.landmark_history = deque(maxlen=5)
        self.fps_history = deque(maxlen=30)

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

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        if not detection_result.face_landmarks:
            return rgb_image

        annotated_image = np.copy(rgb_image)
        height, width = annotated_image.shape[:2]

        for face_landmarks in detection_result.face_landmarks:
            smoothed_landmarks = self.smooth_landmarks(face_landmarks)

            def draw_polyline(index_list, color, thickness):
                for i in range(len(index_list) - 1):
                    pt1 = (int(smoothed_landmarks[index_list[i]].x * width),
                           int(smoothed_landmarks[index_list[i]].y * height))
                    pt2 = (int(smoothed_landmarks[index_list[i+1]].x * width),
                           int(smoothed_landmarks[index_list[i+1]].y * height))
                    cv2.line(annotated_image, pt1, pt2, color, thickness)

            face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                         365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
                         132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
            left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                        159, 160, 161, 246, 33]
            right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388,
                         387, 386, 385, 384, 398, 362]
            lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]  # More stable path
            left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70]
            right_eyebrow = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336]

            draw_polyline(face_oval, (0, 255, 0), 1)
            draw_polyline(left_eye, (255, 0, 0), 1)
            draw_polyline(right_eye, (255, 0, 0), 1)
            draw_polyline(lips, (0, 0, 255), 2)
            draw_polyline(left_eyebrow, (255, 255, 0), 1)
            draw_polyline(right_eyebrow, (255, 255, 0), 1)

        return annotated_image

    def calculate_fps(self, start_time):
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        return np.mean(self.fps_history)

    def draw_info_overlay(self, image, fps, detection_result):
        height = image.shape[0]
        cv2.putText(image, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        face_count = len(detection_result.face_landmarks) if detection_result.face_landmarks else 0
        cv2.putText(image, f'Faces: {face_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if hasattr(detection_result, 'face_blendshapes') and detection_result.face_blendshapes:
            cv2.putText(image, f'Blendshapes: {len(detection_result.face_blendshapes[0])}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if hasattr(detection_result, 'facial_transformation_matrixes') and detection_result.facial_transformation_matrixes:
            cv2.putText(image, 'Transform: Available', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if detection_result.face_landmarks:
            cv2.putText(image, 'Face detected', (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(image, 'Press ESC or Q to quit', (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, 'Press B for blendshapes', (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Starting live face tracking...")

        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            try:
                detection_result = self.detector.detect(mp_image)
                annotated_image = self.draw_landmarks_on_image(rgb_frame, detection_result)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Detection error: {e}")
                annotated_image = frame
                class DummyDetectionResult:
                    def __init__(self):
                        self.face_landmarks = []
                        self.face_blendshapes = []
                        self.facial_transformation_matrixes = []
                detection_result = DummyDetectionResult()

            fps = self.calculate_fps(start_time)
            self.draw_info_overlay(annotated_image, fps, detection_result)
            cv2.imshow('Live Face Tracking', annotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break
            elif key == ord('b'):
                if hasattr(detection_result, 'face_blendshapes') and detection_result.face_blendshapes:
                    print("\n=== Face Blendshapes ===")
                    for blendshape in detection_result.face_blendshapes[0][:10]:
                        print(f"{blendshape.category_name}: {blendshape.score:.4f}")
                    print("========================\n")
            elif key == ord('t'):
                if hasattr(detection_result, 'facial_transformation_matrixes') and detection_result.facial_transformation_matrixes:
                    print("\n=== Facial Transformation Matrix ===")
                    print(detection_result.facial_transformation_matrixes[0])
                    print("====================================\n")

        cap.release()
        cv2.destroyAllWindows()
        print("Face tracking stopped")

def main():
    try:
        tracker = LiveFaceTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
