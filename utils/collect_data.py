import os
import cv2
import time
import uuid
import argparse
import numpy as np
import re
from pathlib import Path
from datetime import datetime

def create_directories():
    """Create necessary directories for data collection"""
    base_dir = os.path.join('data', 'images')
    if not os.path.exists(base_dir):
        print(f"Creating directory: {base_dir}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        print(f"Directory already exists: {base_dir}")
    return base_dir

def get_next_index(behavior_dir, behavior):
    """Find the highest index in existing files and return next index"""
    max_index = -1
    if os.path.exists(behavior_dir):
        for filename in os.listdir(behavior_dir):
            if filename.startswith(f"{behavior}_"):
                # Extract the index from filename (last number before .jpg)
                match = re.search(r'_(\d+)\.jpg$', filename)
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
    
    return max_index + 1  # Return next index after the maximum found

def collect_data(behavior, base_dir, num_images=100, delay=2.0):
    """Collect images for a specific behavior category"""
    # Create behavior-specific directory
    behavior_dir = os.path.join(base_dir, behavior)
    if not os.path.exists(behavior_dir):
        print(f"Creating directory: {behavior_dir}")
        os.makedirs(behavior_dir, exist_ok=True)
    else:
        print(f"Directory already exists: {behavior_dir}")
    
    # Get the next index to start from
    start_index = get_next_index(behavior_dir, behavior)
    print(f"Starting collection from index: {start_index}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not open webcam for {behavior}")
        return
    
    print(f"\nCollecting data for: {behavior}")
    print(f"Press 'q' to quit, 'p' to pause/resume")
    print(f"Images will be saved to: {behavior_dir}")
    print(f"Delay between captures: {delay} seconds")
    
    count = 0
    paused = False
    window_name = f'Collecting {behavior}'
    
    while count < num_images:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Display frame - with status info for preview only
            display_frame = frame.copy()
            status_text = f"Collecting {behavior} - {count+1}/{num_images}"
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.destroyAllWindows()  # Close any previous windows
            cv2.imshow(window_name, display_frame)
            
            # Current index = starting index + count
            current_index = start_index + count
            
            # Save image without any text annotations
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{behavior}_{timestamp}_{current_index}.jpg'
            filepath = os.path.join(behavior_dir, filename)
            cv2.imwrite(filepath, frame)  # Save original frame without text
            
            print(f"Saved image {count+1}/{num_images}: {filename} (index: {current_index})")
            count += 1
            
            # Simple delay between captures
            print(f"Waiting {delay} seconds before next capture...")
            time.sleep(delay)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
            
            # Update the window title to show paused status (display only)
            if paused:
                paused_frame = display_frame.copy()
                cv2.putText(paused_frame, "PAUSED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, paused_frame)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished collecting {count} images for {behavior}")
    print(f"Last saved index: {start_index + count - 1}")

def main():
    # Define available behaviors
    behaviors = ['working', 'slacking']
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Collect training data for slacking detection')
    parser.add_argument('--behavior', type=str, choices=behaviors, required=True,
                      help='Behavior category to collect data for')
    parser.add_argument('--num-images', type=int, default=100,
                      help='Number of images to collect (default: 100)')
    parser.add_argument('--delay', type=float, default=2.0,
                      help='Delay between captures in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Create directories
    base_dir = create_directories()
    
    # Collect data for specified behavior
    collect_data(args.behavior, base_dir, args.num_images, args.delay)

if __name__ == '__main__':
    main() 