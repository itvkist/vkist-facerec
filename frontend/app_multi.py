from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import base64
import json                    
import time
import threading

import os
from PIL import Image

from utils.service.TFLiteFaceAlignment import * 
from utils.service.TFLiteFaceDetector import * 
from utils.functions import *

app = Flask(__name__)

# path = "/home/vkist1/frontend_facerec_VKIST/"
path = "./"

fd_0 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fd_1 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fd_2 = UltraLightFaceDetecion(path + "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fa = CoordinateAlignmentModel(path + "utils/service/weights/coor_2d106.tflite")

url = 'http://localhost:5002/'
# url = 'https://dohubapps.com/user/itvkist/5000/'

api_list = [url + 'facerec', url + 'FaceRec_DREAM', url + 'FaceRec_3DFaceModeling', url + 'check_pickup']
request_times = [15, 10, 10]
api_index = 0
extend_pixel = 50
crop_image_size = 100

recognized_count = 0

# mamnontuoitho 123456789
# secret_key = "506282dd-9dd9-446f-b3f4-0767e0a4b856"

# test vkist
secret_key = "3916a4f1-9069-4f19-ae60-add2f8fdbdb3"

# vkist_6 123456789
# secret_key = "13971a9f-1b2d-46bb-b829-d395431448fd"

predict_labels = []

# gửi ảnh đến một API nhận diện khuôn mặt và xử lý kết quả trả về
def face_recognize(frame, camera):
    global api_index

    # frame là khung hình (ảnh) được chuyển đổi thành định dạng JPEG.
    _, encimg = cv2.imencode(".jpg", frame)
    # img_byte là dạng bytes của ảnh JPEG.
    img_byte = encimg.tobytes()
    # img_str là chuỗi base64 được mã hóa từ img_byte.
    img_str = base64.b64encode(img_byte).decode('utf-8')
    # new_img_str là chuỗi base64 được ghép với định dạng data:image/jpeg;base64, để chuẩn bị gửi lên API.
    new_img_str = "data:image/jpeg;base64," + img_str
    
    ### Thiết lập headers và payload cho yêu cầu POST
    # headers thiết lập kiểu dữ liệu gửi và nhận.
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain', 'charset': 'utf-8'}
    # payload là dữ liệu JSON chứa secret_key và ảnh đã mã hóa.
    payload = json.dumps({"secret_key": secret_key, "img": new_img_str, 'local_register' : False})

    # Gửi yêu cầu POST đến API
    response = requests.post(api_list[api_index], data=payload, headers=headers, timeout=100)

    # Xử lý kết quả trả về từ API
    try:
        for id, name, profileID, timestamp in zip( 
                                                                                        response.json()['result']['id'],
                                                                                        response.json()['result']['identities'],
                                                                                        response.json()['result']['profilefaceIDs'],
                                                                                        response.json()['result']['timelines']
                                                                                        # các trường dữ liệu trả về từ API
                                                                                        ):
            print('Server response', response.json()['result']['identities'])
            if id != -1:
                cur_profile_face = None # lưu ảnh profile và mã hóa thành chuỗi base64
                cur_picker_profile_face = None

                if profileID is not None:
                    cur_url = url + 'images/' + secret_key + '/' + profileID
                    cur_profile_face = np.array(Image.open(requests.get(cur_url, stream=True).raw))
                    cur_profile_face = cv2.cvtColor(cur_profile_face, cv2.COLOR_BGR2RGB)

                    _, encimg = cv2.imencode(".jpg", cur_profile_face)
                    img_byte = encimg.tobytes()
                    img_str = base64.b64encode(img_byte).decode('utf-8')
                    cur_profile_face = "data:image/jpeg;base64," + img_str

                # frame được resize và mã hóa thành chuỗi base64
                frame = cv2.resize(frame, (crop_image_size, crop_image_size))
                _, encimg = cv2.imencode(".jpg", frame)
                img_byte = encimg.tobytes()
                img_str = base64.b64encode(img_byte).decode('utf-8')
                new_img_str = "data:image/jpeg;base64," + img_str

                predict_labels.append([id, name, new_img_str, cur_profile_face, timestamp, camera])

    except requests.exceptions.RequestException:
        print(response.text)


# Mở luồng video từ camera.
# Lấy khung hình từ luồng video, thực hiện một số xử lý như lật ảnh và thay đổi kích thước.
# Nhận diện khuôn mặt và vẽ khung bao quanh các khuôn mặt được nhận diện.
# Định kỳ gửi các khung hình chứa khuôn mặt đến hàm face_recognize để nhận diện.
# Tính toán và hiển thị FPS.
# Chuyển đổi khung hình thành JPEG và trả về dưới dạng bytes để hiển thị trực tuyến.

def get_frame_0():
    # Open the webcam stream
    # ip_cam = 'rtsp://admin:vkist@123@192.168.47.65:554/profile2/media.smp'
    # ip_cam = 'rtsp://admin:Vkist@24@192.168.47.60:554/Streaming/channels/102'
    # ip_cam = 'rtsp://admin:Vkist123@192.168.42.10:554/Streaming/channels/102'
    # ip_cam = 'rtsp://169.254.49.47:554/1/stream1/Profile1'
    # ip_cam = 'rtsp://192.168.47.64/profile2/media.smp'
    # ip_cam = 'rtsp://192.168.47.64:8000/profile2/media.smp'
    
    ip_cam = 'rtsp://admin:vkist@123@192.168.0.101:554/profile2/media.smp'
    # webcam_0 = cv2.VideoCapture(ip_cam)
    webcam_0 = cv2.VideoCapture(0)

    # Lấy kích thước khung hình
    frame_width = int(webcam_0.get(3))
    frame_height = int(webcam_0.get(4))

    # Khai báo biến và danh sách chờ
    prev_frame_time = 0
    new_frame_time = 0
    queue = []
    count = 0

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_0.read()
        orig_image = cv2.flip(orig_image, 1)

        if count % 2 != 0:
            continue
        
        # Sao chép và thay đổi kích thước khung hình
        final_frame = orig_image.copy()
        scale_ratio = 1/1
        new_height, new_width = int(frame_height * scale_ratio), int(frame_width * scale_ratio)
        resized_image = cv2.resize(orig_image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        # Nhận diện khuôn mặt và vẽ khung bao quanh
        temp_resized_boxes, _ = fd_0.inference(resized_image)
        temp_boxes = temp_resized_boxes * (1 / scale_ratio)
        draw_box(final_frame, temp_boxes, color=(125, 255, 125))

        # Tìm landmarks của mỗi khuôn mặt
        temp_resized_marks = fa.get_landmarks(resized_image, temp_resized_boxes)

        # Gửi yêu cầu nhận diện khuôn mặt
        if (count % request_times[api_index]) == 0:
            for bbox_I, landmark_I in zip(temp_resized_boxes, temp_resized_marks):
                landmark_I = landmark_I * (1 / scale_ratio)
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= int(extend_pixel * scale_ratio)
                xmax += int(extend_pixel * scale_ratio)
                ymin -= int(extend_pixel * scale_ratio)
                ymax += int(extend_pixel * scale_ratio)

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin

                xmax = new_width if xmax >= new_width else xmax
                ymax = new_height if ymax >= new_height else ymax

                resized_face_I = resized_image[ymin:ymax, xmin:xmax]
                rotated_resized_face_I = align_face(resized_face_I, landmark_I[34], landmark_I[88])

                camera = "CAM1"
                count = 0
                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_resized_face_I,camera)))
                    queue[-1].start()

        # Tính toán FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # Thêm tên camera vào khung hình
        camera_name = "CAM1"
        cv2.putText(final_frame, camera_name, (frame_width - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Chuyển đổi khung hình sang định dạng JPEG và trả về dưới dạng bytes
        ret, jpeg = cv2.imencode('.jpg', final_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def get_frame_1():
    ip_cam = 'rtsp://admin:vkist@123@192.168.0.102:554/profile2/media.smp'
    # Open the webcam stream
    webcam_1 = cv2.VideoCapture(ip_cam)

    frame_width = int(webcam_1.get(3))
    frame_height = int(webcam_1.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []
    count = 0

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_1.read()
        orig_image = cv2.flip(orig_image, 0)

        if count % 2 != 0:
            continue
        final_frame = orig_image.copy()
        scale_ratio = 1/1
        new_height, new_width = int(frame_height * scale_ratio), int(frame_width * scale_ratio)

        resized_image = cv2.resize(orig_image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        temp_resized_boxes, _ = fd_1.inference(resized_image)

        
        temp_boxes = temp_resized_boxes * (1 / scale_ratio)

        # Draw boundary boxes around faces
        draw_box(final_frame, temp_boxes, color=(125, 255, 125))

        # Find landmarks of each face
        temp_resized_marks = fa.get_landmarks(resized_image, temp_resized_boxes)

        if (count % request_times[api_index]) == 0:
            for bbox_I, landmark_I in zip(temp_resized_boxes, temp_resized_marks):
                landmark_I = landmark_I * (1 / scale_ratio)
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= int(extend_pixel * scale_ratio)
                xmax += int(extend_pixel * scale_ratio)
                ymin -= int(extend_pixel * scale_ratio)
                ymax += int(extend_pixel * scale_ratio)

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = new_width if xmax >= new_width else xmax
                ymax = new_height if ymax >= new_height else ymax

                resized_face_I = resized_image[ymin:ymax, xmin:xmax]
                rotated_resized_face_I = align_face(resized_face_I, landmark_I[34], landmark_I[88])

                count = 0
                queue = [t for t in queue if t.is_alive()]

                camera = "CAM2"
                if len(queue) < 3:
                    # cv2.imwrite('rotated_faces/' + str(time.time()) + '.jpg', rotated_resized_face_I)
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_resized_face_I,camera)))
                    queue[-1].start()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # Add camera name to the opposite corner
        camera_name = "CAM2"
        cv2.putText(final_frame, camera_name, (frame_width - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert the frame to a jpeg image
        ret, jpeg = cv2.imencode('.jpg', final_frame)

        # Return the image as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def get_frame_2():
    # Open the webcam stream
    # ip_cam = 'rtsp://admin:Vkist123@192.168.42.10:554/Streaming/channels/102'
    ip_cam = 'rtsp://admin:vkist@123@192.168.0.103:554/profile2/media.smp'
    webcam_0 = cv2.VideoCapture(ip_cam)
    # webcam_0 = cv2.VideoCapture(0)

    frame_width = int(webcam_0.get(3))
    frame_height = int(webcam_0.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []
    count = 0

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_0.read()
        orig_image = cv2.flip(orig_image, 0)

        final_frame = orig_image.copy()
        scale_ratio = 1/1
        new_height, new_width = int(frame_height * scale_ratio), int(frame_width * scale_ratio)

        resized_image = cv2.resize(orig_image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        temp_resized_boxes, _ = fd_2.inference(resized_image)
        
        temp_boxes = temp_resized_boxes * (1 / scale_ratio)

        # Draw boundary boxes around faces
        draw_box(final_frame, temp_boxes, color=(125, 255, 125))

        # Find landmarks of each face
        temp_resized_marks = fa.get_landmarks(resized_image, temp_resized_boxes)

        if (count % request_times[api_index]) == 0:
            for bbox_I, landmark_I in zip(temp_resized_boxes, temp_resized_marks):
                landmark_I = landmark_I * (1 / scale_ratio)
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= int(extend_pixel * scale_ratio)
                xmax += int(extend_pixel * scale_ratio)
                ymin -= int(extend_pixel * scale_ratio)
                ymax += int(extend_pixel * scale_ratio)

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin

                xmax = new_width if xmax >= new_width else xmax
                ymax = new_height if ymax >= new_height else ymax

                resized_face_I = resized_image[ymin:ymax, xmin:xmax]
                rotated_resized_face_I = align_face(resized_face_I, landmark_I[34], landmark_I[88])

                count = 0
                queue = [t for t in queue if t.is_alive()]

                camera = "CAM3"
                if len(queue) < 3:
                    # cv2.imwrite('rotated_faces/' + str(time.time()) + '.jpg', rotated_resized_face_I)
                    queue.append(threading.Thread(target=face_recognize, args=(rotated_resized_face_I,camera)))
                    queue[-1].start()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        cv2.putText(final_frame, '{0} fps'.format(fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # Add camera name to the opposite corner
        camera_name = "CAM3"
        cv2.putText(final_frame, camera_name, (frame_width - 150, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert the frame to a jpeg image
        ret, jpeg = cv2.imencode('.jpg', final_frame)

        # Return the image as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')

# trường hợp chạy bthg
@app.route('/video_feed_0')
def video_feed_0():
    return Response(get_frame_0(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

# trường hợp chạy test
# @app.route('/video_feed_0')
# def video_feed_0():
#     folder_path = './enhance/vkist306/light'  # Đặt đường dẫn đến folder ảnh
#     return Response(get_frames_from_folder(folder_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_1')
def video_feed_1():
    return Response(get_frame_1(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_2')
def video_feed_2():
    return Response(get_frame_2(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

def get_data():
    while True:
        # Return the image as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/data')
def data():
    global predict_labels
    if len(predict_labels) > 3:
        predict_labels = predict_labels[-3:]
    newest_data = list(reversed(predict_labels))
    # bthg
    return jsonify({'info': newest_data})
    # test
    # return jsonify({'info': newest_data, 'count': recognized_count})

if __name__ == '__main__':
    app.run(debug=True)
