import cv2
import numpy as np
import datetime
from twilio.rest import Client
import time

# Twilio credentials
TWILIO_ACCOUNT_SID = 'AC6aa4772235e9a89e09d3ede5a31a74fe'
TWILIO_AUTH_TOKEN = '2b911af13a55a992ee846540cb4cb23b'
TWILIO_PHONE_NUMBER = '+12512558598'
TO_PHONE_NUMBER = '+918459817144'

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to send SMS alert
def send_sms_alert(message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=TO_PHONE_NUMBER
    )

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables
first_frame = None
motion_detected = False
snapshot_counter = 0
last_sms_time = time.time()  # Initialize the time when the last SMS was sent

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Initialize the first frame
    if first_frame is None:
        first_frame = gray
        continue
    
    # Compute the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_frame, gray)
    
    # Threshold the delta image
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Perform a series of dilations to fill in the holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Minimum contour area to avoid small movements
            continue
        
        # Draw a rectangle around the detected motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Motion detected flag
        motion_detected = True
    
    # If motion is detected, save a snapshot and send SMS
    if motion_detected:
        current_time = time.time()
        
        # Check if a minute has passed since the last SMS
        if current_time - last_sms_time >= 60:  # 60 seconds
            snapshot_counter += 1
            snapshot_filename = f"snapshot_{snapshot_counter}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snapshot_filename, frame)
            print(f"Motion detected! Snapshot saved as {snapshot_filename}")
            
            # Send SMS alert
            send_sms_alert(f"Motion detected! Snapshot saved as {snapshot_filename}")
            
            # Update the time of the last SMS
            last_sms_time = current_time
        
        motion_detected = False  # Reset flag
    
    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
