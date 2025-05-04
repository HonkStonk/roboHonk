from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2

# Import the IMX500 helper class and controls
from picamera2.devices.imx500 import IMX500
from libcamera import (
    controls as libcamera_controls,
)  # Import libcamera controls for accessing metadata
import cv2  # Import OpenCV for drawing
import numpy as np  # Import numpy for image manipulation
import io
from PIL import (
    Image,
)  # Still useful for initial frame handling, although we'll use OpenCV drawing
import hwDef
from packSensor import packSensor
from servoController import ServoController
import time  # Often useful for timing or delays if needed

app = Flask(__name__)

# --- Object Detection Setup ---
# Define the path to the pre-compiled IMX500 MobileNet SSD model
IMX500_MODEL_PATH = (
    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)
# Define the path to the COCO labels file
# This is where you copied the labels file from the examples
COCO_LABELS_PATH = "/home/mow/vsGit/roboHonk/coco_labels.txt"

# Initialize the IMX500 helper class with the model
# This needs to happen after Picamera2 initialization but before starting the camera
# We will initialize it later in the setup after picam2 config.
imx500 = None

# Load the labels for the COCO dataset
labels = []
try:
    with open(COCO_LABELS_PATH, "r") as f:
        for line in f:
            labels.append(line.strip())
    print(f"Loaded {len(labels)} labels from {COCO_LABELS_PATH}")
except FileNotFoundError:
    print(
        f"Error: Labels file not found at {COCO_LABELS_PATH}. Object labels will not be displayed."
    )
    labels = [f"Class {i}" for i in range(91)]  # Fallback labels

# Confidence threshold for displaying detections
CONFIDENCE_THRESHOLD = 0.5  # You can adjust this value


# Initialize camera, battery, servo, and motors
picam2 = Picamera2()

# Configure Picamera2 for the AI Camera, enabling the IMX500 processing pipeline
# We use the raw configuration to access the full sensor data and metadata
# The IMX500 post-processing is handled by the firmware and driver,
# and the results appear in the metadata.
# Using the preview configuration might still work depending on how the IMX500
# integrates, but raw capture is often used when directly accessing metadata.
# Let's start with your existing preview config and see if the metadata is available.
# If not, we might need to switch to a raw configuration.
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},
    # Add the controls to request the CNN output tensor in the metadata
    # The exact control name might vary; check libcamera/picamera2 documentation
    # The documentation you provided mentioned controls::rpi::CnnOutputTensor
    # Let's try accessing it via the metadata object directly.
    # controls={"FrameRate": 60, "rpi.CnnOutputTensor": True}, # This control might not be set directly in config
    controls={"FrameRate": 60},  # Keep your existing controls
    raw={
        "size": (1920, 1080)
    },  # Request raw stream to potentially access more metadata
)
picam2.configure(config)

# Initialize the IMX500 helper after picam2 config
try:
    imx500 = IMX500(IMX500_MODEL_PATH)
    print(f"IMX500 helper initialized with model {IMX500_MODEL_PATH}")
    # It's good practice to set the inference aspect ratio to match the model
    # imx500.set_inference_aspect_ratio(imx500.get_input_size()) # Optional, can help with ROI
except Exception as e:
    print(f"Error initializing IMX500 helper: {e}")
    imx500 = None  # Set to None if initialization fails


picam2.start()

bat = packSensor()
servos = ServoController()

# Initialize motors to stop
left_motor_speed = hwDef.motorStop
right_motor_speed = hwDef.motorStop
servos.leftMotor(left_motor_speed)
servos.rightMotor(right_motor_speed)

# Camera position state
camera_position = 0.5  # Start in middle position (0.0 = down, 1.0 = up)
servos.camServo(camera_position)

motor_base_speed = hwDef.motorMinFwd
steering_offset = 0.0
movement_state = "stop"
MAX_MOTOR_SPEED = 1.0


def generate_frames():
    """
    Captures frames from the camera, performs object detection,
    draws bounding boxes, and yields the annotated frame.
    """
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        metadata = request.get_metadata()

        try:
            annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error converting color format (BGRA2BGR failed): {e}")
            annotated_frame = frame.copy()

        if imx500 is not None:
            try:
                outputs = imx500.get_outputs(metadata)

                print(
                    f"IMX500 Outputs Type: {type(outputs)}, Length: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}"
                )

                if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:

                    # Prints for types and lengths of elements within the list/tuple (keep for now)
                    print(
                        f"Outputs[0] Type: {type(outputs[0])}, Length: {len(outputs[0]) if hasattr(outputs[0], '__len__') else 'N/A'}"
                    )
                    print(
                        f"Outputs[1] Type: {type(outputs[1])}, Length: {len(outputs[1]) if hasattr(outputs[1], '__len__') else 'N/A'}"
                    )
                    print(
                        f"Outputs[2] Type: {type(outputs[2])}, Length: {len(outputs[2]) if hasattr(outputs[2], '__len__') else 'N/A'}"
                    )
                    if len(outputs) > 3:
                        print(
                            f"Outputs[3] Type: {type(outputs[3])}, Length: {len(outputs[3]) if hasattr(outputs[3], '__len__') else 'N/A'}"
                        )

                    if (
                        len(outputs) > 2
                        and isinstance(outputs[0], np.ndarray)
                        and isinstance(outputs[1], np.ndarray)
                        and isinstance(outputs[2], np.ndarray)
                    ):

                        boxes = outputs[0]
                        classes = outputs[1]
                        scores = outputs[2]

                        print(
                            f"Scores Array: Type={type(scores)}, Shape={scores.shape}, Dtype={scores.dtype}"
                        )
                        if isinstance(scores, np.ndarray) and scores.size > 0:
                            print(
                                f"Scores Content (first 10): {scores[:10]}, Max Score: {scores.max() if scores.size > 0 else 'N/A'}, Min Score: {scores.min() if scores.size > 0 else 'N/A'}"
                            )
                        else:
                            print("Scores Array is empty or not a numpy array.")

                        print(
                            f"Boxes Array: Type={type(boxes)}, Shape={boxes.shape}, Dtype={boxes.dtype}"
                        )
                        print(
                            f"Classes Array: Type={type(classes)}, Shape={classes.shape}, Dtype={classes.dtype}"
                        )

                        if (
                            isinstance(scores, np.ndarray)
                            and scores.size > 0
                            and isinstance(boxes, np.ndarray)
                            and isinstance(classes, np.ndarray)
                            and scores.shape[0] == boxes.shape[0]
                            and scores.shape[0] == classes.shape[0]
                        ):

                            img_height, img_width, _ = annotated_frame.shape
                            detections_count = scores.shape[0]
                            print(f"--- Detections Found: {detections_count} ---")

                            for i in range(detections_count):
                                confidence = scores[i]

                                # Scale the confidence score from 0-100 to 0.0-1.0
                                # Based on observed Max Score values around 80-90.
                                scaled_confidence = confidence / 100.0

                                # print(f"Detection {i}: Raw Confidence={confidence:.2f}, Scaled Confidence={scaled_confidence:.2f}") # Debug print for confidence

                                # Only process detections above the confidence threshold (using the scaled score)
                                if scaled_confidence > CONFIDENCE_THRESHOLD:
                                    if i < boxes.shape[0]:
                                        ymin, xmin, ymax, xmax = boxes[i]

                                        coords_xywh = (
                                            xmin,
                                            ymin,
                                            xmax - xmin,
                                            ymax - ymin,
                                        )
                                        scaled_coords = imx500.convert_inference_coords(
                                            coords_xywh, metadata, picam2
                                        )

                                        # --- FIX: Unpack the tuple returned by convert_inference_coords ---
                                        # The error indicates it returns a tuple (x, y, w, h).
                                        if (
                                            isinstance(scaled_coords, tuple)
                                            and len(scaled_coords) == 4
                                        ):
                                            x, y, w, h = (
                                                scaled_coords  # Unpack the tuple directly
                                            )
                                            # print(f"Scaled Coords (unpacked for detection {i}): x={x}, y={y}, w={w}, h={h}") # Debug print
                                        else:
                                            # If it's not the expected tuple, print a warning and skip drawing for this detection
                                            print(
                                                f"Warning: imx500.convert_inference_coords returned unexpected type for detection {i}: {type(scaled_coords)}"
                                            )
                                            continue  # Skip to the next detection

                                        # Ensure coordinates are valid integers and within frame boundaries
                                        x, y, w, h = int(x), int(y), int(w), int(h)
                                        x = max(0, x)
                                        y = max(0, y)
                                        draw_w = max(0, min(w, img_width - x))
                                        draw_h = max(0, min(h, img_height - y))

                                        if draw_w > 0 and draw_h > 0:
                                            label = (
                                                labels[int(classes[i])]
                                                if 0 <= int(classes[i]) < len(labels)
                                                else f"Class {int(classes[i])}"
                                            )
                                            label_text = f"{label}: {scaled_confidence:.2f}"  # Use scaled confidence in label
                                            color = (0, 255, 0)  # Green color (BGR)

                                            cv2.rectangle(
                                                annotated_frame,
                                                (x, y),
                                                (x + draw_w, y + draw_h),
                                                color,
                                                2,
                                            )

                                            text_x = max(0, x)
                                            text_y = max(15, y - 10)
                                            cv2.putText(
                                                annotated_frame,
                                                label_text,
                                                (text_x, text_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                2.7,
                                                color,
                                                2,
                                            )
                                        # else: print invalid dims warning (optional)

                                    # else: print boxes index warning (optional)
                            # else: print below threshold warning (optional)
                            # else: print inconsistent counts warning (optional)

                        else:
                            print(
                                "--- No Detections (scores array empty) or Inconsistent Output Counts ---"
                            )
                            # You could optionally print shapes here for debugging
                            # if isinstance(scores, np.ndarray): print(f"Shapes: scores={scores.shape}, boxes={boxes.shape if isinstance(boxes, np.ndarray) else 'N/A'}, classes={classes.shape if isinstance(classes, np.ndarray) else 'N/A'}")

                    else:
                        print(
                            "--- Outputs[0], Outputs[1], or Outputs[2] are not NumPy Arrays as expected ---"
                        )
                        # You could optionally print types here for debugging
                        # if isinstance(outputs, list) and len(outputs) > 2: print(f"Types: outputs[0]={type(outputs[0])}, outputs[1]={type(outputs[1])}, outputs[2]={type(outputs[2])}")

                elif outputs is None:
                    print("--- IMX500 Outputs is None ---")

                else:
                    print(
                        f"--- Outputs structure is unexpected. Type: {type(outputs)}, Length: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'} ---"
                    )

            except Exception as e:
                print(f"Error processing object detection outputs: {e}")

        request.release()

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            print("Error encoding frame to JPEG")
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    """Route to provide the video stream with object detection."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    """Render the main control page."""
    return render_template("index.html")


@app.route("/battery")
def battery_status():
    """Return the current battery percentage."""
    soc = bat.getPackSOC()
    return jsonify({"battery_percentage": soc})


@app.route("/move_servo", methods=["POST"])
def move_servo():
    """Handle requests to move the camera servo."""
    global camera_position
    direction = request.json.get("direction")
    if direction == "up":
        camera_position = min(camera_position + 0.05, 1.0)
    elif direction == "down":
        camera_position = max(camera_position - 0.05, 0.0)
    servos.camServo(camera_position)
    return jsonify({"camera_position": camera_position})


@app.route("/control_motors", methods=["POST"])
def control_motors():
    """Handle requests to control the robot motors."""
    global left_motor_speed, right_motor_speed
    global movement_state, steering_offset, motor_base_speed

    action = request.json.get("action")
    STEERING_STEP = 0.02
    # Ensure motors stay <= 1.0 when adjusting for steering
    STEERING_LIMIT = MAX_MOTOR_SPEED - hwDef.motorMinFwd

    if action == "forward":
        movement_state = "forward"
        steering_offset = 0.0  # negative = left, positive = right
        motor_base_speed = hwDef.motorMinFwd

    elif action == "backward":
        movement_state = "backward"
        steering_offset = 0.0
        left_motor_speed = hwDef.motorMinBack
        right_motor_speed = hwDef.motorMinBack

    elif action == "stop":
        movement_state = "stop"
        steering_offset = 0.0
        left_motor_speed = hwDef.motorStop
        right_motor_speed = hwDef.motorStop

    elif action == "left":
        if movement_state == "forward":
            steering_offset = max(steering_offset - STEERING_STEP, -STEERING_LIMIT)
        # In-place turning when stopped
        elif movement_state == "stop":
            left_motor_speed = hwDef.motorMinBack
            right_motor_speed = hwDef.motorMinFwd

    elif action == "right":
        if movement_state == "forward":
            steering_offset = min(steering_offset + STEERING_STEP, STEERING_LIMIT)
        # In-place turning when stopped
        elif movement_state == "stop":
            left_motor_speed = hwDef.motorMinFwd
            right_motor_speed = hwDef.motorMinBack

    # Apply differential drive when moving forward
    if movement_state == "forward":
        left_motor_speed = max(
            hwDef.motorMinFwd, min(MAX_MOTOR_SPEED, motor_base_speed + steering_offset)
        )
        right_motor_speed = max(
            hwDef.motorMinFwd, min(MAX_MOTOR_SPEED, motor_base_speed - steering_offset)
        )

    # Update motors - ensure speeds are within acceptable range for your servo controller
    servos.leftMotor(min(left_motor_speed, MAX_MOTOR_SPEED))
    servos.rightMotor(min(right_motor_speed, MAX_MOTOR_SPEED))

    return jsonify(
        {
            "left_motor_speed": left_motor_speed,
            "right_motor_speed": right_motor_speed,
            "steering_offset": steering_offset,
            "movement_state": movement_state,
        }
    )


if __name__ == "__main__":
    # Make sure OpenCV is installed: pip install opencv-python
    # Make sure picamera2 and its dependencies are installed
    # Make sure imx500-all and imx500-tools are installed
    app.run(host="0.0.0.0", port=5000, threaded=True)
