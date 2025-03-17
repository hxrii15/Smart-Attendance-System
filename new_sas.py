import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Create photos directory if it doesn't exist
if not os.path.exists("photos"):
    os.makedirs("photos")
    print("Created 'photos' directory. Please add reference images there.")
    exit()

# Check if any images exist in photos directory
photo_files = [f for f in os.listdir("photos") if f.endswith(('.jpg', '.jpeg', '.png'))]
if not photo_files:
    print("Error: No image files found in the 'photos' folder")
    print("Please add photos of students to the 'photos' folder")
    exit()

print("Loading face detector and recognizer...")
# Use OpenCV's built-in face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Use LBPH Face Recognizer (built into OpenCV)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Extract names from file names (removing numbers)
known_names = []
for filename in photo_files:
    # Extract name from filename (removing extension and numbers)
    name = os.path.splitext(filename)[0]
    # Remove digits from the name
    name = ''.join([c for c in name if not c.isdigit()])
    if name not in known_names:
        known_names.append(name)

print(f"Found potential students: {known_names}")

# Prepare training data
faces = []
labels = []
label_names = {}
current_label = 0

print("Training face recognizer...")
# Process each photo for training
for name in known_names:
    # Find all files that start with this name
    name_files = [f for f in photo_files if f.startswith(name)]
    if not name_files:
        continue
        
    print(f"Processing photos for {name}...")
    label_names[current_label] = name
    
    for filename in name_files:
        # Load image file
        image_path = os.path.join("photos", filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not load image {filename}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in detected_faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            # Resize to a standard size (helps recognition)
            face_roi = cv2.resize(face_roi, (100, 100))
            # Add to training data
            faces.append(face_roi)
            labels.append(current_label)
            print(f"Added face from {filename}")
            
    if any(label == current_label for label in labels):
        current_label += 1
    else:
        print(f"Warning: No faces found in photos for {name}")

if not faces:
    print("Error: No faces could be detected in any of the provided photos")
    print("Please ensure the photos contain clear, front-facing faces")
    exit()

# Train the recognizer
face_recognizer.train(faces, np.array(labels))
print(f"Trained recognizer with {len(faces)} faces from {len(known_names)} students")

# Current date for CSV filename
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create CSV file for attendance if it doesn't exist
csv_path = current_date + ".csv"
attendance_marked = set()  # To track which students have already been marked

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])
else:
    # Load already marked attendance to avoid duplicates
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Ensure row is not empty
                attendance_marked.add(row[0])

print("Starting video capture...")
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture device")
    exit()

print("Attendance system running. Press 'q' to quit.")

# Process frames at a lower rate for better performance
frame_count = 0
process_every = 3  # Process every 3rd frame

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    
    while True:
        # Read a frame from the video stream
        ret, frame = video_capture.read()
        
        if not ret:
            print("Failed to grab frame from camera")
            break
        
        # Only process every nth frame to improve performance
        frame_count += 1
        if frame_count % process_every != 0:
            # Still display the frame with any previous detections
            cv2.imshow('Attendance System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            # Resize to same size used in training
            face_roi = cv2.resize(face_roi, (100, 100))
            
            try:
                # Predict label and confidence
                label, confidence = face_recognizer.predict(face_roi)
                
                # Lower confidence value means better match in LBPH
                # Set a threshold for acceptable confidence
                if confidence < 70:  # Adjust this threshold based on testing
                    name = label_names[label]
                    display_text = f"{name} ({confidence:.1f})"
                    color = (0, 255, 0)  # Green for recognized
                    
                    # Mark attendance if not already marked
                    if name not in attendance_marked:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        writer.writerow([name, current_time])
                        f.flush()  # Make sure it's written to file immediately
                        attendance_marked.add(name)
                        print(f"Marked attendance for {name} at {current_time}")
                else:
                    display_text = f"Unknown ({confidence:.1f})"
                    color = (0, 0, 255)  # Red for unknown
            except:
                display_text = "Error"
                color = (0, 0, 255)  # Red
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display name and confidence
            cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, display_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show attendance status if recognized
            if display_text != "Unknown" and display_text != "Error":
                status = "Marked" if name in attendance_marked else "Not Marked"
                cv2.putText(frame, status, (x+5, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Automatic attendance system", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Attendance System', frame)
        
        # If 'q' is pressed, quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()

try:
    # Sort the attendance CSV file by name
    sorted_rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get the header
        sorted_rows = sorted(list(reader), key=lambda row: row[0].lower())  # Sort by name alphabetically

    # Write sorted rows back to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        writer.writerows(sorted_rows)  # Write sorted rows

    print(f"Attendance for {current_date} has been saved to {csv_path} in alphabetical order")
except Exception as e:
    print(f"Error sorting attendance data: {str(e)}")