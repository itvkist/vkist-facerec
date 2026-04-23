import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # suppress oneDNN numerical warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress TF C++ INFO/WARNING logs

from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import base64
import json
import time
import threading
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from utils.service.TFLiteFaceAlignment import *
from utils.service.TFLiteFaceDetector import *
from utils.functions import *

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Backend URL — change to domain name if not running locally
BACKEND_URL = 'http://localhost:5002/'

# Account secret key from the backend (id: abc / password: 123)
SECRET_KEY = "281f8065-1719-4670-8f39-ef7686c03902"

# Send a recognition request every N frames
REQUEST_INTERVAL = 15

# Pixels to extend around detected face crop before sending to backend
EXTEND_PIXEL = 50

# Max concurrent recognition threads per stream
MAX_QUEUE = 3

# RTSP stream URLs — add as many as needed, leave empty to use webcams only
RTSP_URLS = [
    # 'rtsp://admin:@Vkist308@10.3.8.118/media/video1',
    # 'rtsp://admin:@Vkist308@10.3.8.126/media/video1',
]

# ── Model initialization ───────────────────────────────────────────────────────

WEIGHTS_PATH = "./utils/service/weights/"
fa = CoordinateAlignmentModel(WEIGHTS_PATH + "coor_2d106.tflite")

# ── Shared state ───────────────────────────────────────────────────────────────

predict_labels = []
predict_labels_lock = threading.Lock()

# ── Helpers ────────────────────────────────────────────────────────────────────

def image_to_base64(frame):
    """Encode a BGR numpy frame to a base64 JPEG data URI."""
    _, encimg = cv2.imencode(".jpg", frame)
    img_str = base64.b64encode(encimg.tobytes()).decode('utf-8')
    return "data:image/jpeg;base64," + img_str


def fetch_profile_image(profile_id):
    """Fetch a profile face image from the backend and return as base64 data URI."""
    if profile_id is None:
        return None
    url = BACKEND_URL + 'images/' + SECRET_KEY + '/' + profile_id
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image_to_base64(img)


def face_recognize(frame):
    """Send a face crop to the backend for recognition and store results."""
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}
    payload = json.dumps({
        "secret_key": SECRET_KEY,
        "img": image_to_base64(frame),
        "local_register": False
    })

    try:
        response = requests.post(
            BACKEND_URL + 'facerec',
            data=payload,
            headers=headers,
            timeout=100
        )
        result = response.json()['result']

        for person_id, name, description, profile_id, timestamp in zip(
            result['id'],
            result['identities'],
            result['descriptions'],
            result['profilefaceIDs'],
            result['timelines']
        ):
            if person_id == -1:
                continue

            face_thumb = image_to_base64(cv2.resize(frame, (100, 100)))
            profile_face = fetch_profile_image(profile_id)

            with predict_labels_lock:
                predict_labels.append([person_id, name, face_thumb, profile_face, timestamp, description])

    except requests.exceptions.RequestException as e:
        print(f'Recognition request failed: {e}')


def open_camera(source):
    """
    Open a camera source. Returns (cap, flip_code) or raises RuntimeError.
    flip_code: 1 = horizontal flip (webcam), -1 = no flip (RTSP/IP cam).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")
    # Flip horizontally for webcams (mirror effect), not for RTSP
    flip_code = 1 if isinstance(source, int) else None
    return cap, flip_code


def detect_webcams():
    """Try webcam indices 0 and 1, return list of available indices."""
    available = []
    for index in range(2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(index)
        cap.release()
    return available


def stream_frames(source):
    """
    Generator that yields MJPEG frames from a camera source (webcam index or RTSP URL).
    Runs face detection every REQUEST_INTERVAL frames and spawns recognition threads.
    Each stream gets its own face detector instance to avoid race conditions.
    """
    fd = UltraLightFaceDetecion(WEIGHTS_PATH + "RFB-320.tflite", conf_threshold=0.98)

    cap, flip_code = open_camera(source)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    prev_frame_time = 0
    queue = []
    count = 0

    while True:
        ret, orig_image = cap.read()
        if not ret:
            print(f'Stream lost: {source}')
            break

        # Mirror webcam frames horizontally
        if flip_code is not None:
            orig_image = cv2.flip(orig_image, flip_code)

        final_frame = orig_image.copy()
        count += 1

        # ── Face detection ─────────────────────────────────────────────────
        boxes, _ = fd.inference(orig_image)
        draw_box(final_frame, boxes, color=(125, 255, 125))
        landmarks = fa.get_landmarks(orig_image, boxes)

        # ── Send crops to backend every REQUEST_INTERVAL frames ────────────
        if (count % REQUEST_INTERVAL) == 0:
            count = 0
            queue = [t for t in queue if t.is_alive()]

            for bbox, landmark in zip(boxes, landmarks):
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Expand bounding box slightly for better recognition
                xmin = max(0, xmin - EXTEND_PIXEL)
                ymin = max(0, ymin - EXTEND_PIXEL)
                xmax = min(frame_width,  xmax + EXTEND_PIXEL)
                ymax = min(frame_height, ymax + EXTEND_PIXEL)

                face_crop = orig_image[ymin:ymax, xmin:xmax]
                aligned_face = align_face(face_crop, landmark[34], landmark[88])

                if len(queue) < MAX_QUEUE:
                    t = threading.Thread(target=face_recognize, args=(aligned_face,))
                    t.start()
                    queue.append(t)

        # ── FPS overlay ────────────────────────────────────────────────────
        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time else 0
        prev_frame_time = new_frame_time
        cv2.putText(final_frame, f'{fps} fps', (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # ── Encode and yield MJPEG frame ───────────────────────────────────
        _, jpeg = cv2.imencode('.jpg', final_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()


# ── Stream registry ────────────────────────────────────────────────────────────
# Build the list of all active streams: available webcams + configured RTSP URLs

def build_stream_sources():
    sources = detect_webcams()
    sources += RTSP_URLS
    return sources

STREAM_SOURCES = build_stream_sources()
print(f'Active streams: {STREAM_SOURCES}')


# ── Flask routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', stream_count=len(STREAM_SOURCES))


@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    """Stream MJPEG video for the given stream index."""
    if stream_id >= len(STREAM_SOURCES):
        return 'Stream not found', 404
    return Response(
        stream_frames(STREAM_SOURCES[stream_id]),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/data')
def data():
    """Return the latest recognition results (last 3 entries)."""
    with predict_labels_lock:
        newest = list(reversed(predict_labels[-3:]))
    return jsonify({'info': newest})


if __name__ == '__main__':
    app.run(debug=True)
