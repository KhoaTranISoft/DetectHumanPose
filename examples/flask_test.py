from flask import Flask, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture("data\cauthang.mp4") # Sử dụng camera mặc định

def generate_frames():
    while True:
        success, frame = camera.read() # Đọc frame từ camera
        if not success:
            break
        else:
            # Chuyển đổi frame sang dạng JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Đẩy frame đến Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
