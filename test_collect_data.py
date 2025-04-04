import os
import cv2
import time
import uuid
import argparse
import numpy as np
from pathlib import Path

def main():
    
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 