from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2
from PIL import Image
import io
import hwDef
from packSensor import packSensor
from servoController import ServoController

app = Flask(__name__)

# Initialize camera, battery, servo, and motors
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},
    controls={"FrameRate": 60},
)
picam2.configure(config)
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

motor_base_speed = hwDef.motorMinFwd
steering_offset = 0.0
movement_state = "stop"
STEERING_STEP = 0.01
MAX_MOTOR_SPEED = 1.0


def generate_frames():
    while True:
        frame = picam2.capture_array()
        image = Image.fromarray(frame).convert("RGB")
        stream = io.BytesIO()
        image.save(stream, format="JPEG")
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + stream.getvalue() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/battery")
def battery_status():
    soc = bat.getPackSOC()
    return jsonify({"battery_percentage": soc})


@app.route("/move_servo", methods=["POST"])
def move_servo():
    global camera_position
    direction = request.json.get("direction")
    if direction == "up":
        camera_position = min(camera_position + 0.1, 1.0)
    elif direction == "down":
        camera_position = max(camera_position - 0.1, 0.0)
    servos.camServo(camera_position)
    return jsonify({"camera_position": camera_position})


@app.route("/control_motors", methods=["POST"])
def control_motors():
    global left_motor_speed, right_motor_speed
    global movement_state, steering_offset, motor_base_speed

    action = request.json.get("action")
    STEERING_STEP = 0.03
    STEERING_LIMIT = 1.0 - hwDef.motorMinFwd  # Ensure motors stay <= 1.0

    if action == "forward":
        movement_state = "forward"
        steering_offset = -0.05
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

    elif action == "right":
        if movement_state == "forward":
            steering_offset = min(steering_offset + STEERING_STEP, STEERING_LIMIT)

    # Apply differential drive when moving forward
    if movement_state == "forward":
        left_motor_speed = motor_base_speed + max(0.0, steering_offset)
        right_motor_speed = motor_base_speed + max(0.0, -steering_offset)

    # In-place turning when stopped
    elif movement_state == "stop":
        if action == "left":
            left_motor_speed = hwDef.motorMinBack
            right_motor_speed = hwDef.motorMinFwd
        elif action == "right":
            left_motor_speed = hwDef.motorMinFwd
            right_motor_speed = hwDef.motorMinBack

    # Update motors
    servos.leftMotor(min(left_motor_speed, 1.0))
    servos.rightMotor(min(right_motor_speed, 1.0))

    return jsonify(
        {
            "left_motor_speed": left_motor_speed,
            "right_motor_speed": right_motor_speed,
            "steering_offset": steering_offset,
            "movement_state": movement_state,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
