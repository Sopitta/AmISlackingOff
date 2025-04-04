import cv2
import numpy as np
import time
import threading
import winsound  # For Windows alarm sound
from ultralytics import YOLO

def play_alarm(duration=3000):
    """Play an alarm sound for the specified duration in milliseconds."""
    frequency = 2500  # Set frequency in Hz
    winsound.Beep(frequency, duration)

def test_model():
    # Load your trained model
    model = YOLO('runs/detect/train2/weights/best.pt')  # adjust path if needed
    print("Model loaded successfully!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Define class names (adjust according to your data.yaml)
    class_names = ['Slacking off', 'Working']
    
    # Variables for slacking detection
    slacking_count = 0
    alarm_active = False
    alarm_thread = None
    alarm_end_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Make detections 
        results = model(frame, conf=0.25)  # adjust confidence threshold if needed
        
        is_slacking = False
        current_status = "No Detection"
        
        for r in results:
            boxes = r.boxes
            if len(boxes) == 0:
                continue
            
            # Find the box with maximum confidence
            max_conf_box = max(boxes, key=lambda b: b.conf.item())
            
            # Get box coordinates
            x1, y1, x2, y2 = max_conf_box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class name and confidence
            cls = int(max_conf_box.cls.item())
            conf = float(max_conf_box.conf.item())
            
            # Draw box and label
            color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # Red for slacking, Green for working
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            current_status = class_names[cls]
            #label = f'{current_status}: {conf:.2f}'
            label = f'{current_status}'
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Update slacking detection
            is_slacking = (cls == 0)  # Slacking class index is 0
        
        # Update slacking counter
        if is_slacking:
            slacking_count += 1
        else:
            slacking_count = 0
        
        # Check if alarm should be triggered
        current_time = time.time()
        if slacking_count >= 20 and not alarm_active:
            # Trigger alarm
            alarm_active = True
            alarm_end_time = current_time + 3  # Alarm for 3 seconds
            # Start alarm in a separate thread to avoid blocking the video feed
            if alarm_thread is None or not alarm_thread.is_alive():
                alarm_thread = threading.Thread(target=play_alarm)
                alarm_thread.daemon = True
                alarm_thread.start()
        
        # Check if alarm should be stopped
        if alarm_active and current_time > alarm_end_time:
            alarm_active = False
        
        # Display alarm status and slacking counter
        #alarm_status = "ALARM ACTIVE" if alarm_active else "No Alarm"
        alarm_status = "Go back to work!!" if alarm_active else "Everything is fine :)"
        cv2.putText(frame, f"Status: {current_status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #cv2.putText(frame, f"Slacking frames: {slacking_count}/20", (10, 60), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{alarm_status}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if alarm_active else (255, 255, 255), 2)
        
        # If alarm is active, flash the screen border red
        if alarm_active:
            height, width = frame.shape[:2]
            thickness = 20
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness)
        
        # Display the frame
        cv2.imshow('Slacking Detector', frame)
        
        # Break loop with 'q' or reset alarm with 'r'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            alarm_active = False
            slacking_count = 0
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()