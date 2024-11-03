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
    action = request.json.get("action")

    if action == "forward":
        left_motor_speed = hwDef.motorMinFwd
        right_motor_speed = hwDef.motorMinFwd
    elif action == "backward":
        left_motor_speed = hwDef.motorMinBack
        right_motor_speed = hwDef.motorMinBack
    elif action == "left":
        left_motor_speed = hwDef.motorMinBack  # Left motor backward
        right_motor_speed = hwDef.motorMinFwd  # Right motor forward
    elif action == "right":
        left_motor_speed = hwDef.motorMinFwd  # Left motor forward
        right_motor_speed = hwDef.motorMinBack  # Right motor backward
    elif action == "stop":
        left_motor_speed = hwDef.motorStop
        right_motor_speed = hwDef.motorStop

    servos.leftMotor(left_motor_speed)
    servos.rightMotor(right_motor_speed)
    return jsonify(
        {"left_motor_speed": left_motor_speed, "right_motor_speed": right_motor_speed}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
