import cv2

# Virtual camera index: 0, 1, or higher depending on your system
camera_index = 1
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not access the OBS Virtual Camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("OBS Stream - iPhone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q'
        break

cap.release()
cv2.destroyAllWindows()
