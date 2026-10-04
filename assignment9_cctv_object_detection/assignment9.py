# ============================================================
# AIML Assignment 9
# Object Detection from CCTV Footage
# Using YOLO and OpenCV
# ============================================================

import cv2
from ultralytics import YOLO


# ------------------------------------------------------------
# 1. Load YOLO Model
# ------------------------------------------------------------

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")


# ------------------------------------------------------------
# 2. Input and Output Video
# ------------------------------------------------------------

input_video = "CCTV_video.mp4"

output_video = "output_detection.mp4"


# ------------------------------------------------------------
# 3. Open CCTV Video
# ------------------------------------------------------------

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():

    print("ERROR: Could not open CCTV video.")
    print("Make sure CCTV_video.mp4 is inside this folder.")

    exit()


# ------------------------------------------------------------
# 4. Get Video Properties
# ------------------------------------------------------------

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps == 0:

    fps = 25


# ------------------------------------------------------------
# 5. Create Output Video
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ------------------------------------------------------------
# 6. Process Video
# ------------------------------------------------------------

print("\nStarting object detection...")
print("Press Q to stop the program.\n")


frame_count = 0


while True:

    # Read one frame
    ret, frame = cap.read()

    # Stop when video ends
    if not ret:

        break


    frame_count += 1


    # --------------------------------------------------------
    # Perform Object Detection
    # --------------------------------------------------------

    results = model(
        frame,
        verbose=False
    )


    # --------------------------------------------------------
    # Draw Bounding Boxes
    # --------------------------------------------------------

    annotated_frame = results[0].plot()


    # --------------------------------------------------------
    # Display Frame
    # --------------------------------------------------------

    cv2.imshow(
        "CCTV Object Detection",
        annotated_frame
    )


    # --------------------------------------------------------
    # Save Frame
    # --------------------------------------------------------

    out.write(
        annotated_frame
    )


    # --------------------------------------------------------
    # Press Q to Stop
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ------------------------------------------------------------
# 7. Release Resources
# ------------------------------------------------------------

cap.release()

out.release()

cv2.destroyAllWindows()


# ------------------------------------------------------------
# 8. Final Message
# ------------------------------------------------------------

print("\n===================================")
print("OBJECT DETECTION COMPLETED")
print("===================================")

print("Total frames processed:", frame_count)

print("Output video:", output_video)