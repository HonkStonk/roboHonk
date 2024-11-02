from flask import Flask, Response
from picamera2 import Picamera2
import io
from PIL import Image

app = Flask(__name__)

# Initialize the camera once
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},  # Set resolution here (width, height)
    controls={"FrameRate": 60},  # Set frame rate here
)
picam2.configure(config)
picam2.start()


def generate_frames():
    while True:
        frame = picam2.capture_array()  # Capture an array
        image = Image.fromarray(frame).convert("RGB")  # Convert to RGB
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
