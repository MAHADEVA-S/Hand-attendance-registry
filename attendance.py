import logging
import time
from typing import Optional, List
from datetime import datetime
import cv2
import sqlite3
import mediapipe as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HandTracker:
    def __init__(self):
        """Initialize hand tracking with Mediapipe"""
        logger.info("Initializing HandTracker...")
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=4,  # Increased to handle more users
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.colors = [
                (0, 255, 0),  # Green
                (0, 0, 255),   # Red
                (255, 0, 0),   # Blue
                (255, 255, 0)  # Yellow
            ]
            logger.info("Hand tracker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing HandTracker: {e}")
            raise

    def detect_hands(self, frame) -> dict:
        """Detect and differentiate hands in the given frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_data = {
            'landmarks': [],
            'handedness': [],
            'colors': []
        }
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_data['landmarks'].append(hand_landmarks)
                
                # Get handedness (left/right)
                if results.multi_handedness:
                    handedness = results.multi_handedness[i].classification[0].label
                    hand_data['handedness'].append(handedness)
                else:
                    hand_data['handedness'].append('Unknown')
                
                # Assign color based on index
                hand_data['colors'].append(self.colors[i % len(self.colors)])
                
                logger.debug(f"Hand {i} detected: {hand_data['handedness'][-1]}")
        
        return hand_data

    def draw_hands(self, frame, hand_data, user_ids=None):
        """Draw hands with user differentiation"""
        if not hand_data['landmarks']:
            return frame
            
        for i, landmarks in enumerate(hand_data['landmarks']):
            # Draw landmarks with assigned color
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=hand_data['colors'][i], thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(
                    color=hand_data['colors'][i], thickness=2, circle_radius=2)
            )
            
            # Add user ID if available
            if user_ids and i < len(user_ids):
                x = int(landmarks.landmark[0].x * frame.shape[1])
                y = int(landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(frame, f"User: {user_ids[i]}", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_data['colors'][i], 2)

class AttendanceSystem:
    def __init__(self):
        """Initialize attendance tracking system"""
        from database import Database
        self.db = Database()
        self.hand_tracker = HandTracker()
        self.last_attendance: Optional[datetime] = None

    def get_user_id(self) -> str:
        """Generate a unique user ID in ESF7001 format"""
        return self.db.get_next_user_id()

    def mark_attendance(self, hand_data) -> list:
        """Mark attendance for multiple users"""
        now = datetime.now()
        results = []
        
        if self.last_attendance is None or (now - self.last_attendance).total_seconds() > 5:
            for landmarks in hand_data['landmarks']:
                try:
                    # Extract and serialize landmark data
                    serialized_landmarks = []
                    for landmark in landmarks.landmark:
                        serialized_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    # Identify or register user
                    user_id = self.db.identify_user(serialized_landmarks)
                    if not user_id:
                        user_id = self.get_user_id()
                        self.db.register_user(user_id, serialized_landmarks)
                    
                    # Mark attendance
                    if self.db.mark_attendance(user_id):
                        self.db.update_user_last_seen(user_id)
                        results.append(user_id)
                        logger.info(f"Attendance marked for user {user_id} at {now}")
                except Exception as e:
                    logger.error(f"Attendance error: {e}")
                    continue
            
            if results:
                self.last_attendance = now
                
        return results

    def run(self):
        """Run the attendance tracking system"""
        logger.info("Starting attendance system")
        # Verify OpenCV installation and backend
        logger.info("Checking OpenCV installation...")
        try:
            # Get OpenCV version
            version = cv2.__version__
            logger.info(f"OpenCV version: {version}")
            
            # Use default OpenCV logging
            logger.info("Using default OpenCV logging")
            
            # Check video capture support
            if not hasattr(cv2, 'VideoCapture'):
                logger.error("OpenCV video capture not supported")
                return
                
            # Check basic image reading
            if not hasattr(cv2, 'imread'):
                logger.error("OpenCV image reading not supported")
                return
        except Exception as e:
            logger.error(f"OpenCV verification failed: {e}")
            return
        
        # Initialize camera with enhanced error handling
        cap = None
        max_attempts = 3
        camera_backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_ANY      # Auto-detect backend
        ]
        
        for backend in camera_backends:
            for camera_index in [1, 0]:  # Try both common indices
                for attempt in range(max_attempts):
                    try:
                        logger.info(f"Attempting to open camera index {camera_index} with backend {backend} (attempt {attempt+1})")
                        cap = cv2.VideoCapture(camera_index, backend)
                        
                        if cap.isOpened():
                            # Set basic camera properties
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                            
                            # Verify camera feed with frame analysis
                            valid_frames = 0
                            for _ in range(10):  # Check more frames
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    # Check frame validity
                                    mean_brightness = cv2.mean(frame)[0]
                                    if mean_brightness > 10:  # Check average brightness
                                        valid_frames += 1
                                        if valid_frames >= 3:  # Need multiple good frames
                                            logger.info(f"Camera successfully opened (index {camera_index}, backend {backend})")
                                            logger.info(f"Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
                                            logger.info(f"Mean brightness: {mean_brightness:.1f}")
                                            break
                            else:
                                logger.warning(f"Camera opened but failed to capture valid frames (mean brightness: {mean_brightness:.1f})")
                                cap.release()
                                continue
                            break
                        else:
                            logger.warning(f"Failed to open camera index {camera_index} with backend {backend}")
                            continue
                    except Exception as e:
                        logger.error(f"Camera initialization error: {e}")
                        if cap:
                            cap.release()
                        continue
                    
                    time.sleep(1)
                else:
                    continue
                break
            else:
                continue
            break
        else:
            logger.error("Could not initialize any camera after multiple attempts")
            cv2.destroyAllWindows()
            return

        try:
            while cap.isOpened():
                # Read frame with validation
                max_frame_retries = 3
                frame = None
                valid_frame = False
                
                for attempt in range(max_frame_retries):
                    success, frame = cap.read()
                    
                    # Validate frame dimensions and content
                    if success and frame is not None:
                        height, width, _ = frame.shape
                        if height > 0 and width > 0:
                            valid_frame = True
                            break
                    
                    logger.debug(f"Frame read attempt {attempt+1} failed")
                    time.sleep(0.1)
                
                if not valid_frame:
                    logger.warning("Could not read valid frame after multiple attempts")
                    # Check if camera is still open
                    if not cap.isOpened():
                        logger.error("Camera connection lost")
                        break
                    continue

                frame = cv2.flip(frame, 1)
                hand_data = self.hand_tracker.detect_hands(frame)

                if hand_data['landmarks']:
                    # Show user count
                    user_count = self.db.get_user_count()
                    cv2.putText(frame, f"Users: {user_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Check for attendance gesture (any wrist above threshold)
                    attendance_gesture = False
                    for landmarks in hand_data['landmarks']:
                        wrist_y = landmarks.landmark[self.hand_tracker.mp_hands.HandLandmark.WRIST].y
                        if wrist_y < 0.4:
                            attendance_gesture = True
                            break

                    if attendance_gesture:
                        # Mark attendance for all users
                        user_ids = self.mark_attendance(hand_data)
                        if user_ids:
                            self.hand_tracker.draw_hands(frame, hand_data, user_ids)
                            cv2.putText(frame, "Attendance Marked!", (50, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, "Thank You!", (50, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # Add 2 second delay after marking attendance
                            cv2.waitKey(2000)
                        else:
                            self.hand_tracker.draw_hands(frame, hand_data)
                    else:
                        self.hand_tracker.draw_hands(frame, hand_data)

                cv2.imshow('Hand Tracking Attendance', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    logger.info("User requested shutdown")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            logger.info("System shutdown complete")

if __name__ == "__main__":
    try:
        system = AttendanceSystem()
        system.run()
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("Exiting...")
