import cv2

# Load the pre-trained people detection cascade
people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Access OBS Virtual Camera
camera_index = 1
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not access the OBS Virtual Camera.")
    exit()

# Counter for detected people
people_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale (Haar cascades work with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the frame
    people = people_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    # Draw rectangles around detected people and update count
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        people_count += 1

    # Display the frame with detections
    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("People Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Total People Detected: {people_count}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
