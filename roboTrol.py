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
        request_obj = picam2.capture_request()
        frame = request_obj.make_array(
            "main"
        )  # This is typically RGB or YUV from Picamera2
        metadata = request_obj.get_metadata()

        # Picamera2 make_array("main") often gives RGB.
        # OpenCV drawing functions (rectangle, putText) expect BGR by default unless color is specified as RGB.
        # cv2.imencode expects BGR or BGRA.
        # If 'frame' is RGB, convert to BGR for OpenCV drawing and encoding.
        annotated_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if imx500 is not None:
            try:
                # Call get_outputs with add_batch=True as seen in the demo script
                outputs = imx500.get_outputs(metadata, add_batch=True)

                if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                    # Based on the demo script, the outputs are structured as:
                    # outputs[0]: boxes ([1, N, 4] with ymin, xmin, ymax, xmax)
                    # outputs[1]: scores ([1, N])
                    # outputs[2]: classes ([1, N]) -> These should be integer class IDs

                    # Extract the tensors, taking the first element [0] for the single image in the batch
                    raw_boxes_tensor = outputs[0][0]  # Shape [N, 4]
                    scores_tensor = outputs[1][0]  # Shape [N]
                    classes_tensor = outputs[2][
                        0
                    ]  # Shape [N] - Expected to be integer class IDs

                    if (
                        isinstance(raw_boxes_tensor, np.ndarray)
                        and isinstance(scores_tensor, np.ndarray)
                        and isinstance(classes_tensor, np.ndarray)
                    ):
                        # Determine the actual number of detections from the scores tensor length
                        actual_num_detections = scores_tensor.shape[0]

                        img_height, img_width, _ = annotated_frame_bgr.shape

                        for i in range(actual_num_detections):
                            scaled_confidence = scores_tensor[
                                i
                            ]  # Scores are already 0.0-1.0

                            # Only process and draw high-confidence items
                            if scaled_confidence > CONFIDENCE_THRESHOLD:

                                # --- Class ID Interpretation (CORRECTED) ---
                                # Use the integer class ID directly from the classes_tensor
                                current_class_id = int(
                                    classes_tensor[i]
                                )  # Ensure it's an integer

                                # Determine the label name using the correct integer class ID
                                final_label_name = "Unknown"
                                if 0 <= current_class_id < len(labels):
                                    final_label_name = labels[current_class_id]
                                else:
                                    final_label_name = f"ClassID_OOB_{current_class_id}"  # Out of bounds

                                print(
                                    f"\n--- DETECTED (idx {i}, score {scaled_confidence:.2f} > {CONFIDENCE_THRESHOLD}) ---"
                                )
                                print(
                                    f"  Class Info: ClassID={current_class_id} ('{final_label_name}')"
                                )

                                # Get raw box coordinates [ymin, xmin, ymax, xmax]
                                # These are usually normalized (0.0 to 1.0) by the model
                                ymin_raw, xmin_raw, ymax_raw, xmax_raw = (
                                    raw_boxes_tensor[i]
                                )
                                print(
                                    f"  Raw Box [ymin,xmin,ymax,xmax]: [{ymin_raw:.4f}, {xmin_raw:.4f}, {ymax_raw:.4f}, {xmax_raw:.4f}]"
                                )

                                # Check for invalid raw box dimensions
                                if xmin_raw >= xmax_raw or ymin_raw >= ymax_raw:
                                    print(
                                        f"  WARNING: Invalid raw box dimensions (min >= max). Skipping this detection."
                                    )
                                    continue

                                # The imx500.convert_inference_coords function expects (x, y, w, h)
                                # If raw boxes are (ymin, xmin, ymax, xmax), convert them:
                                box_norm_x = xmin_raw
                                box_norm_y = ymin_raw
                                box_norm_w = xmax_raw - xmin_raw
                                box_norm_h = ymax_raw - ymin_raw

                                coords_xywh_normalized = (
                                    box_norm_x,
                                    box_norm_y,
                                    box_norm_w,
                                    box_norm_h,
                                )
                                print(
                                    f"  Normalized Box (x,y,w,h) for conversion: [{coords_xywh_normalized[0]:.4f}, {coords_xywh_normalized[1]:.4f}, {coords_xywh_normalized[2]:.4f}, {coords_xywh_normalized[3]:.4f}]"
                                )

                                # Check for invalid normalized box dimensions before scaling
                                if box_norm_w <= 0 or box_norm_h <= 0:
                                    print(
                                        f"  WARNING: Normalized width or height is <= 0. This might result in zero drawable dimensions. W={box_norm_w:.4f}, H={box_norm_h:.4f}"
                                    )
                                    # Continue processing to see if convert_inference_coords handles it
                                    # continue # Removed continue here

                                scaled_coords_xywh = imx500.convert_inference_coords(
                                    coords_xywh_normalized, metadata, picam2
                                )
                                print(
                                    f"  Scaled Coords (x,y,w,h) from convert_inference_coords: {scaled_coords_xywh}"
                                )

                                if (
                                    isinstance(scaled_coords_xywh, tuple)
                                    and len(scaled_coords_xywh) == 4
                                ):
                                    x, y, w, h = (
                                        scaled_coords_xywh  # These should be pixel values for drawing
                                    )
                                    x, y, w, h = (
                                        int(x),
                                        int(y),
                                        int(w),
                                        int(h),
                                    )  # Ensure they are integers

                                    # Clamp coordinates to be within frame boundaries for drawing
                                    draw_x = max(0, x)
                                    draw_y = max(0, y)
                                    # Adjust width/height if x/y were clamped, to prevent drawing outside right/bottom edges
                                    draw_w = max(0, min(w, img_width - draw_x))
                                    draw_h = max(0, min(h, img_height - draw_y))

                                    print(
                                        f"  Drawable Coords (clamped x,y,w,h): [{draw_x}, {draw_y}, {draw_w}, {draw_h}]"
                                    )
                                    # Removed the "Skipping drawing" printout as drawing is now always attempted.

                                    # Use the correct label name and scaled confidence
                                    label_text = (
                                        f"{final_label_name}: {scaled_confidence:.2f}"
                                    )
                                    color_bgr = (
                                        0,
                                        255,
                                        0,
                                    )  # Green in BGR for OpenCV

                                    # Attempt to draw the rectangle and text regardless of draw_w/draw_h being zero
                                    cv2.rectangle(
                                        annotated_frame_bgr,
                                        (draw_x, draw_y),
                                        (draw_x + draw_w, draw_y + draw_h),
                                        color_bgr,
                                        2,
                                    )
                                    text_y_pos = max(15, draw_y - 10)
                                    cv2.putText(
                                        annotated_frame_bgr,
                                        label_text,
                                        (draw_x, text_y_pos),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        color_bgr,
                                        2,
                                    )

                                else:
                                    print(
                                        f"  Warning: imx500.convert_inference_coords returned unexpected type or length: {type(scaled_coords_xywh)}, value: {scaled_coords_xywh}"
                                    )
                    else:  # outputs[0][0], outputs[1][0], or outputs[2][0] are not NumPy arrays
                        print(
                            "--- Expected output tensors (boxes, scores, classes) are not NumPy Arrays ---"
                        )
                elif outputs is None:
                    print("--- IMX500 Outputs is None (possibly intermittent) ---")
                else:  # Outputs structure unexpected
                    print(
                        f"--- Outputs structure is unexpected. Type: {type(outputs)}, Length: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'} ---"
                    )

            except Exception as e:
                print(f"Error processing object detection outputs: {e}")
                import traceback

                traceback.print_exc()

        request_obj.release()

        ret, buffer = cv2.imencode(".jpg", annotated_frame_bgr)  # Encode the BGR frame
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
