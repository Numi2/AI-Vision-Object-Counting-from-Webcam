import ultralytics
ultralytics.checks()

import cv2
from ultralytics import solutions

# Open the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(1)
assert cap.isOpened(), "Error accessing webcam"

# Define points for a line or region of interest in the webcam feed
line_points = [(600, 0), (400, 720)]  # Vertical line at x = 400, from top (y=0) to bottom (y=720)
# Initialize the Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=line_points,
    model="yolo11m.pt",
    line_width=2,
)

# Main loop to read frames from webcam
while True:
    success, im0 = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    # Run object counter
    results = counter(im0)

    # Display the frame
    cv2.imshow("Webcam Object Counter", results.plot_im)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
