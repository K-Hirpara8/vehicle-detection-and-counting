import cv2
from ultralytics import YOLO
import math

model = YOLO("best.pt")
video_path = "456.mp4"

cap = cv2.VideoCapture(video_path)

frame_skip = 1
frame_count = 0
last_results = None

# Two detection lines
line_B = 350   # Upper line
line_A = 600   # Lower line

entered = 0
left = 0

# Store each object's centroid history
objects = {}            # object_id : (cx, cy)
object_id = 0

def match_object(cx, cy, objects):
    for oid, (px, py) in objects.items():
        distance = math.hypot(cx - px, cy - py)
        if distance < 40:          # threshold for being same object
            return oid
    return None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run inference every N frames
    if frame_count % frame_skip == 0:
        results = model(frame, verbose=False)[0]
        last_results = results
    else:
        results = last_results

    new_objects = {}

    if results is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name != "vehicle":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Match to existing object ID
            matched_id = match_object(cx, cy, objects)

            if matched_id is None:
                matched_id = object_id
                object_id += 1

            # Check movement direction
            if matched_id in objects:
                px, py = objects[matched_id]

                # From top - down (entered)
                if py < line_A and cy >= line_A:
                    entered += 1

                # From bottom - up (left)
                if py > line_B and cy <= line_B:
                    left += 1

            new_objects[matched_id] = (cx, cy)

            # Show the ID
            cv2.putText(frame, f"ID:{matched_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    objects = new_objects

    # Draw lines
    cv2.line(frame, (0, line_A), (frame.shape[1], line_A), (255, 0, 0), 2)
    cv2.line(frame, (0, line_B), (frame.shape[1], line_B), (0, 0, 255), 2)

    # Display counts
    cv2.putText(frame, f"Entered: {entered}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.putText(frame, f"Left: {left}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Vehicle Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
