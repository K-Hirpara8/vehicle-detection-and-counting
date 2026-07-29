import cv2
from ultralytics import YOLO
import math

# Load the trained model
model = YOLO(r"runs\detect\runs\train1\weights\best.pt")

# Input video
video_path = r"3.mp4"

cap = cv2.VideoCapture(video_path)

# Read original video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# Create output video
output_path = "vehicle_counting_output.mp4"

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

frame_skip = 1
frame_count = 0
last_results = None

# Detection lines
line_B = 500   # Exit line
line_A = 700   # Entry line

# Vehicle counters
entered = 0
left = 0

# Store already-counted vehicle IDs
counted_entered = set()
counted_left = set()

# Store vehicle positions
objects = {}
object_id = 0


def match_object(cx, cy, objects, used_ids):
    best_id = None
    best_distance = float("inf")

    for oid, (px, py) in objects.items():

        # Do not assign the same old ID to multiple vehicles
        if oid in used_ids:
            continue

        distance = math.hypot(cx - px, cy - py)

        if distance < 40 and distance < best_distance:
            best_distance = distance
            best_id = oid

    return best_id


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run vehicle detection
    if frame_count % frame_skip == 0:
        results = model(frame, verbose=False)[0]
        last_results = results
    else:
        results = last_results

    new_objects = {}

    if results is not None:
        used_ids = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Process only vehicles
            if class_name != "vehicle":
                continue

            # Bounding-box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Calculate vehicle centre
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Match vehicle with an existing ID
            matched_id = match_object(
                cx,
                cy,
                objects,
                used_ids
            )

            # Create a new ID when no match is found
            if matched_id is None:
                matched_id = object_id
                object_id += 1

            used_ids.add(matched_id)

            # Check movement direction
            if matched_id in objects:
                px, py = objects[matched_id]

                # Moving downward across entry line
                if (
                    py < line_A
                    and cy >= line_A
                    and matched_id not in counted_entered
                ):
                    entered += 1
                    counted_entered.add(matched_id)

                # Moving upward across exit line
                if (
                    py > line_B
                    and cy <= line_B
                    and matched_id not in counted_left
                ):
                    left += 1
                    counted_left.add(matched_id)

            # Save current vehicle position
            new_objects[matched_id] = (cx, cy)

            # Display vehicle ID
            cv2.putText(
                frame,
                f"ID:{matched_id}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    objects = new_objects

    # Draw entry line
    cv2.line(
        frame,
        (0, line_A),
        (w, line_A),
        (255, 0, 0),
        3
    )

    cv2.putText(
        frame,
        "ENTRY LINE",
        (w - 200, line_A - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # Draw exit line
    cv2.line(
        frame,
        (0, line_B),
        (w, line_B),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "EXIT LINE",
        (w - 180, line_B - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Create dashboard background
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (390, 130),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    # Dashboard title
    cv2.putText(
        frame,
        "VEHICLE COUNTING SYSTEM",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Entered vehicles
    cv2.putText(
        frame,
        f"Entered: {entered}",
        (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Exited vehicles
    cv2.putText(
        frame,
        f"Exited: {left}",
        (25, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Save processed frame
    out.write(frame)

    # Display processed frame
    cv2.imshow("Vehicle Counting", frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {output_path}")