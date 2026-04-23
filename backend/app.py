# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import math
import base64
import datetime
import traceback
import uuid
from functools import wraps

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
from numpy import dot, sqrt
import cv2
from PIL import Image
import requests
import hnswlib
import face_alignment

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import jinja2
import aiohttp_jinja2
from aiohttp import web
from aiohttp_swagger import *
from pubsub import pub

from sqlalchemy import func, delete
from sqlalchemy.sql import text
from sqlalchemy.orm import aliased

# ── Local ──────────────────────────────────────────────────────────────────────
from app.arcface.backbone import Backbone
from app.vision.ssd.config.fd_config import define_img_size
from app.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from create_app import (db_session, engine, Base, DefineImages, People, Timeline,
                        Users, ChildrenPicker, PeopleClasses, Classes, PickUp, verify_pass)
from deep3d.util.load_mats import load_lm3d
from deep3d.options.test_options import TestOptions
from deep3d.models import create_model
from deep3d.util.visualizer import MyVisualizer
from deep3d.util.preprocess import align_img
from face_dream.dream import Branch, norm_angle


# ══════════════════════════════════════════════════════════════════════════════
# AI MODELS
# ══════════════════════════════════════════════════════════════════════════════

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Face detector (RFB-640)
class_names = [name.strip() for name in open('./app/vision/detect_RFB_640/voc-model-labels.txt').readlines()]
candidate_size = 1000
threshold = 0.7
define_img_size(640)
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=device)
net.load('./app/vision/detect_RFB_640/version-RFB-640.pth')

# ArcFace backbone
input_size = [112, 112]
transform = transforms.Compose([
    transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
    transforms.CenterCrop([input_size[0], input_size[1]]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
backbone = Backbone(input_size)
backbone.load_state_dict(torch.load(
    './app/arcface/ms1mv3_arcface_r50_fp16/backbone_ir50_ms1m_epoch120.pth',
    map_location=torch.device(device)
))
backbone.to(device)
backbone.eval()

# Accessories classifier
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
accessories_classify = torch.load('./app/accessories_classification/shuffle0_0_epoch_47.pth', map_location=device)
accessories_classify.eval()

# Deep3D (3D face reconstruction for mask removal)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', face_detector='blazeface')
opt = TestOptions().parse()
model_deep3d = create_model(opt)
model_deep3d.setup(opt)
model_deep3d.device = device
model_deep3d.parallelize()
model_deep3d.eval()
visualizer = MyVisualizer(opt)
lm3d_std = load_lm3d('./deep3d/BFM')

# Dream embedding (pose-invariant feature correction)
model_dream = Branch(feat_dim=512)
checkpoint = torch.load('./face_dream/checkpoint_512.pth')
model_dream.load_state_dict(checkpoint['state_dict'])
model_dream.eval()


# ══════════════════════════════════════════════════════════════════════════════
# AI FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(x, y):
    return dot(x, y) / (sqrt(dot(x, x)) * sqrt(dot(y, y)))


def no_accent_vietnamese(utf8_str):
    INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d" + \
             "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D"
    r = re.compile("|".join(INTAB))
    replaces_dict = dict(zip(INTAB, OUTTAB))
    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)


def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def load_image(img):
    if type(img).__module__ == np.__name__:
        return img
    if len(img) > 11 and img[0:11] == "data:image/":
        return loadBase64Img(img)
    if len(img) > 11 and img.startswith("http"):
        return np.array(Image.open(requests.get(img, stream=True).raw))
    if not os.path.isfile(img):
        raise ValueError("Confirm that ", img, " exists")
    return cv2.imread(img)


def check_accessories(image):
    """Classify accessories (0=none, 1=mask, 2=unknown)."""
    image_tensor = data_transform(image).unsqueeze_(0)
    input_ = Variable(image_tensor).to(device)
    output = accessories_classify(input_)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities.data.cpu().numpy().argmax()


def detect_landmark(img):
    """Detect 5 facial landmarks from a PIL image."""
    preds = fa.get_landmarks_from_image(np.array(img))[0]
    left_eye_x  = (preds[37][0] + preds[40][0]) / 2
    left_eye_y  = (preds[37][1] + preds[40][1]) / 2
    right_eye_x = (preds[43][0] + preds[46][0]) / 2
    right_eye_y = (preds[43][1] + preds[46][1]) / 2
    nose_x      = (preds[30][0] + preds[33][0]) / 2
    nose_y      = (preds[30][1] + preds[33][1]) / 2
    return np.array([
        [left_eye_x,    left_eye_y],
        [right_eye_x,   right_eye_y],
        [nose_x,        nose_y],
        [preds[48][0],  preds[48][1]],
        [preds[54][0],  preds[54][1]],
    ], dtype='f')


def reconstruct(im, lm):
    """Run Deep3D face reconstruction."""
    W, H = im.size
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    data = {
        'imgs': torch.tensor(np.array(im) / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
        'lms':  torch.tensor(lm).unsqueeze(0)
    }
    model_deep3d.set_input(data)
    model_deep3d.test()


def rasterize():
    """Render the reconstructed 3D face and return as RGB numpy array."""
    visuals = model_deep3d.get_current_visuals()
    return visualizer.save_img(visuals)


def unmask(input_img):
    """Remove mask from face using 3D reconstruction."""
    lm = detect_landmark(input_img)
    reconstruct(input_img, lm)
    return rasterize()


# Pose estimation
def estimatePose(frame, landmarks):
    """Calculate poses"""
    size = frame.shape #(height, width, color_channel)
    image_points = np.array([
                            (landmarks[30][0], landmarks[30][1]),     # Nose tip
                            (landmarks[8][0], landmarks[8][1]),       # Chin
                            (landmarks[36][0], landmarks[36][1]),     # Left eye left corner
                            (landmarks[45][0], landmarks[45][1]),     # Right eye right corne
                            (landmarks[48][0], landmarks[48][1]),     # Left Mouth corner
                            (landmarks[54][0], landmarks[54][1])      # Right mouth corner
                        ], dtype="double")
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)#, flags=cv2.CV_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    
    return str(int(roll)), str(int(pitch)), str(int(yaw)), image_points


def dream_embedding(embedding_I, input_yaw):
    """Apply DREAM correction to embedding for non-frontal yaw angles."""
    yaw = np.zeros([1, 1])
    yaw[0, 0] = norm_angle(float(input_yaw))
    original_embedding_tensor = np.expand_dims(embedding_I.detach().cpu().numpy(), axis=0)
    feature_original = torch.autograd.Variable(torch.from_numpy(original_embedding_tensor.astype(np.float32)))
    yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)))
    new_embedding = model_dream(feature_original, yaw)
    return new_embedding.cpu().data.numpy()[0, :]


def get_embeddings(img_input, local_register=False):
    """Detect faces and extract embeddings. Returns (feats, images, bboxes, accessories, yaws)."""
    img = load_image(img_input)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not local_register:
        boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])

    feats, images, bboxes, accessories, yaws = [], [], [], [], []

    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = boxes[i]
        xmin -= (xmax - xmin) / 18
        xmax += (xmax - xmin) / 18
        ymin -= (ymax - ymin) / 18
        ymax += (ymax - ymin) / 18
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)
        boxes[i, :] = [xmin, ymin, xmax, ymax]

        infer_img = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        if infer_img is None or infer_img.shape[0] == 0 or infer_img.shape[1] == 0:
            continue

        accessory_id = 2
        with torch.no_grad():
            accessory_id = check_accessories(Image.fromarray(infer_img))
            if accessory_id == 1:
                try:
                    infer_img = unmask(Image.fromarray(np.uint8(infer_img)).convert('RGB'))
                except Exception as e:
                    print(f'unmask failed: {e}')
            feat = F.normalize(backbone(transform(Image.fromarray(infer_img)).unsqueeze(0).to(device))).cpu()

        landmarks = fa.get_landmarks_from_image(infer_img)
        if landmarks is None:
            continue

        roll, pitch, yaw, _ = estimatePose(infer_img, landmarks[0])

        # Apply DREAM correction for non-frontal faces (|yaw| > 15°)
        if abs(int(yaw)) > 15:
            feat = dream_embedding(feat, yaw)

        feats.append(feat.detach().cpu().numpy() if hasattr(feat, 'detach') else feat)
        images.append(infer_img.copy())
        bboxes.append("{} {} {} {}".format(xmin, ymin, xmax, ymax))
        accessories.append(str(accessory_id))
        yaws.append(yaw)

    return feats, images, bboxes, accessories, yaws


def validate_request(req, keys, values):
    """Fill missing keys in req with defaults from values."""
    new_req = {}
    for key in keys:
        new_req[key] = req[key] if key in req else values[key]
    return new_req


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

class LoginApp():
    def __init__(self):
        pass

    def login_required(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = None
            try:
                access_key = args[0].cookies['user_face_key']
            except:
                access_key = None
            if access_key:
                user = Users.query.filter_by(access_key=access_key).first()
                if user:
                    user = user.__dict__
            return f(*((user,) + args), **kwargs)
        return decorated

    def login_user(self, user, request):
        user.access_key = str(uuid.uuid4())
        db_session.commit()
        response = web.HTTPSeeOther(location='./')
        response.cookies['user_face_key'] = user.access_key
        return response

    def logout_user(self, request):
        response = web.HTTPSeeOther(location='./')
        response.cookies['user_face_key'] = ''
        return response


login_app = LoginApp()


@aiohttp_jinja2.template('index.html')
@login_app.login_required
async def index(current_user, request):
    data = await request.post()

    if 'login' in data and request.method == 'POST':
        username = data.get('username')
        password = data.get('password')
        user = Users.query.filter_by(username=username).first()
        if user and verify_pass(password, user.password):
            return login_app.login_user(user, request)
        raise web.HTTPFound(location='./')

    if 'register' in data and request.method == 'POST':
        username = data.get('username')
        password = data.get('password')
        if username.strip() == "" or password.strip() == "":
            raise web.HTTPFound(location='./')
        if Users.query.filter_by(username=username).first():
            raise web.HTTPFound(location='./')
        user = Users(username=username, password=password, secret_key="")
        db_session.add(user)
        db_session.commit()
        p = hnswlib.Index(space='cosine', dim=512)
        p.init_index(max_elements=1000, ef_construction=200, M=16)
        p.set_ef(10)
        p.set_num_threads(4)
        p.save_index("indexes/index_" + str(user.secret_key) + ".bin")
        return login_app.login_user(user, request)

    if not current_user:
        return {'is_login': False, 'current_user': None}
    return {'is_login': True, 'current_user': current_user}


@aiohttp_jinja2.template('client.html')
@login_app.login_required
async def client_a(current_user, request):
    return {'current_user': current_user}


async def logout(request):
    return login_app.logout_user(request)


async def facerec(request):
    """
    ---
    description: This end-point allow to recognize face identity.
    tags:
    - Face Recognition
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền secret key
        "403":
            description: Secret key không hợp lệ
    """
    req = await request.json()

    if 'secret_key' not in req:
        return web.json_response({"result": {'message': 'Vui lòng truyền secret key'}}, status=400)

    user = Users.query.filter_by(secret_key=req['secret_key']).first()
    if not user:
        return web.json_response({"result": {'message': 'Secret key không hợp lệ'}}, status=403)

    img_input = req.get('img', '')
    if not img_input.startswith('data:image/'):
        return web.json_response({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}, status=400)

    feats, images, bboxes, masks, yaws = get_embeddings(img_input)

    p = hnswlib.Index(space='cosine', dim=512)
    p.load_index("indexes/index_" + str(user.secret_key) + '.bin')

    generated_face_ids = []
    profile_face_ids = []
    person_access_keys = []
    identities = []
    descriptions = []
    timelines = []

    for feat, image, mask, yaw in zip(feats, images, masks, yaws):
        person_access_key = -1
        try:
            neighbors, distances = p.knn_query(feat, k=1)
            if distances[0][0] <= 0.45:
                person_access_key = db_session.query(
                    DefineImages.person_access_key,
                    func.count(DefineImages.person_access_key).label('total')
                ).filter(
                    DefineImages.id.in_(neighbors.tolist()[0]),
                    DefineImages.person_access_key == People.access_key,
                    People.user_id == user.id
                ).group_by(DefineImages.person_access_key)\
                 .order_by(text('total DESC')).first().person_access_key
        except:
            person_access_key = -1

        person = People.query.filter_by(access_key=person_access_key).first()
        name = person.name if person else 'Người lạ'
        description = person.description if person else None
        print(f'Detected: {name}')
        identities.append(name)
        descriptions.append(description)
        person_access_keys.append(person_access_key)

        profile_image_id = DefineImages.query.filter_by(person_access_key=person_access_key).first()
        profile_face_ids.append(profile_image_id.image_id if profile_image_id is not None else None)
        generated_face_ids.append(None)

        now = round(datetime.datetime.now().timestamp() * 1000)
        extra = str(uuid.uuid4())
        if not os.path.isdir("images/" + req['secret_key']):
            os.mkdir("images/" + req['secret_key'])
        cv2.imwrite("images/" + req['secret_key'] + "/face_" + str(now) + '_' + extra + ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        entry = Timeline(
            user_id=user.id,
            person_access_key=person_access_key,
            image_id="face_" + str(now) + '_' + extra,
            embedding=np.array2string(feat, separator=','),
            timestamp=now,
            mask=mask,
            yaw=yaw
        )
        db_session.add(entry)
        db_session.commit()
        timelines.append(now)
        pub.sendMessage('face_vkist', uid=req['secret_key'], message='facerec ' + str(now))

    return web.json_response({'result': {
        "bboxes": bboxes,
        "identities": identities,
        "descriptions": descriptions,
        "id": person_access_keys,
        "timelines": timelines,
        "profilefaceIDs": profile_face_ids,
        "3DFace": generated_face_ids,
        "masks": masks
    }}, status=200)


async def facereg(request):
    """
    ---
    description: This end-point allow to enroll face identity.
    tags:
    - Face Registration
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền secret key
        "403":
            description: Secret key không hợp lệ
    """
    req = await request.json()

    if 'secret_key' not in req:
        return web.json_response({"result": {'message': 'Vui lòng truyền secret key'}}, status=400)

    user = Users.query.filter_by(secret_key=req['secret_key']).first()
    if not user:
        return web.json_response({"result": {'message': 'Secret key không hợp lệ'}}, status=400)

    feats, images, boxes = [], [], []
    if "img" in req:
        for img_input in req["img"]:
            if not img_input.startswith('data:image/'):
                return web.json_response({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}, status=400)
            local = 'local_register' in req
            feats_, images_, boxes_, masks_, _ = get_embeddings(img_input, local)
            feats += feats_
            images += images_
            boxes += boxes_

    if len(boxes) < 1 and 'access_key' not in req:
        return web.json_response({"result": {'message': 'Không xác định khuôn mặt'}}, status=400)

    if 'access_key' not in req or req['access_key'] == "":
        if 'img' not in req or 'type_role' not in req or 'name' not in req:
            return web.json_response({"result": {'message': 'Vui lòng tryền đầy đủ dữ liệu'}}, status=400)
        access_key = str(uuid.uuid4())
        req = validate_request(req,
            ['name', 'age', 'type_role', 'class_access_key', 'gender', 'phone', 'description', 'secret_key'],
            {'name': None, 'age': None, 'type_role': None, 'class_access_key': None, 'gender': None, 'phone': None, 'description': None, 'secret_key': None})
        person = People(user_id=user.id, name=req['name'], age=req['age'], type_role=req['type_role'],
                        gender=req['gender'], phone=req['phone'], description=req['description'], access_key=access_key)
        db_session.add(person)
        db_session.commit()
        if not PeopleClasses.query.filter_by(person_access_key=person.access_key).first():
            if req['class_access_key'] is not None and req['type_role'] != 'parent':
                db_session.add(PeopleClasses(class_access_key=req['class_access_key'], person_access_key=person.access_key))
                db_session.commit()
    else:
        access_key = req['access_key']
        person = People.query.filter_by(access_key=access_key, user_id=user.id).first()
        req = validate_request(req,
            ['name', 'age', 'type_role', 'class_access_key', 'gender', 'phone', 'description', 'secret_key'],
            person.__dict__)
        person.name = req['name']
        person.age = req['age']
        person.type_role = req['type_role']
        person.gender = req['gender']
        person.phone = req['phone']
        person.description = req['description']
        db_session.commit()
        pc = PeopleClasses.query.filter_by(person_access_key=person.access_key).first()
        if not pc:
            if req['class_access_key'] is not None and req['type_role'] != 'parent':
                db_session.add(PeopleClasses(class_access_key=req['class_access_key'], person_access_key=person.access_key))
                db_session.commit()
        else:
            pc.class_access_key = req['class_access_key']
            db_session.commit()

    person = People.query.filter_by(access_key=access_key).first()

    if 'applicant_access_keys' in req and req['applicant_access_keys'] is not None and req['type_role'] != "teacher":
        for ak in req['applicant_access_keys']:
            if req['type_role'] == 'student':
                parent = People.query.filter_by(access_key=ak, type_role="parent", user_id=user.id).first()
                if parent:
                    if not ChildrenPicker.query.filter_by(child_access_key=person.access_key, picker_access_key=parent.id).first():
                        db_session.add(ChildrenPicker(child_access_key=person.access_key, picker_access_key=parent.id))
                        db_session.commit()
            else:
                child = People.query.filter_by(access_key=ak, type_role="student", user_id=user.id).first()
                if child:
                    if not ChildrenPicker.query.filter_by(child_access_key=child.id, picker_access_key=person.access_key).first():
                        db_session.add(ChildrenPicker(child_access_key=child.id, picker_access_key=person.access_key))
                        db_session.commit()

    p = hnswlib.Index(space='cosine', dim=512)
    p.load_index("indexes/index_" + str(user.secret_key) + '.bin', max_elements=1000)

    for feat, image in zip(feats, images):
        now = round(datetime.datetime.now().timestamp() * 1000)
        if not os.path.isdir("images/" + req['secret_key']):
            os.mkdir("images/" + req['secret_key'])
        extra = str(uuid.uuid4())
        cv2.imwrite("images/" + req['secret_key'] + "/face_" + str(now) + '_' + extra + ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        entry = Timeline(user_id=user.id, person_access_key=person.access_key,
                         image_id="face_" + str(now) + '_' + extra,
                         embedding=np.array2string(feat, separator=','), timestamp=now)
        db_session.add(entry)
        db_session.commit()
        define_image = DefineImages(person_access_key=person.access_key, image_id=entry.image_id)
        db_session.add(define_image)
        db_session.commit()
        define_image = DefineImages.query.filter_by(image_id=entry.image_id).first()
        p.add_items(feat, np.array([define_image.id]))
        p.save_index("indexes/index_" + str(user.secret_key) + '.bin')

    pub.sendMessage('face_vkist', uid=user.secret_key,
                    message='facereg ' + str(datetime.datetime.now().timestamp() * 1000))

    return web.json_response({"result": {'message': 'success', 'access_key': access_key}}, status=200)


@login_app.login_required
async def delete_image(current_user, request):
    """
    ---
    description: This end-point allow to delete face image.
    tags:
    - Face Image Deleting
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền access key
        "403":
            description: Bạn không có quyền xóa
    """
    req = await request.json()
    if 'access_key' not in req:
        return web.json_response({"result": {'message': 'Vui lòng truyền access key'}}, status=400)

    person = People.query.filter_by(access_key=req['access_key'], user_id=current_user['id']).first()
    if not person:
        return web.json_response({"result": {'message': 'Bạn không có quyền xóa'}}, status=403)

    ak = person.access_key
    db_session.execute(delete(DefineImages).where(DefineImages.person_access_key == ak))
    db_session.execute(delete(ChildrenPicker).where(ChildrenPicker.child_access_key == ak))
    db_session.execute(delete(ChildrenPicker).where(ChildrenPicker.picker_access_key == ak))
    db_session.execute(delete(PeopleClasses).where(PeopleClasses.person_access_key == ak))
    db_session.execute(delete(PickUp).where(PickUp.child_access_key == ak))
    db_session.execute(delete(PickUp).where(PickUp.picker_access_key == ak))
    db_session.delete(person)
    db_session.commit()

    p = hnswlib.Index(space='cosine', dim=512)
    p.init_index(max_elements=1000, ef_construction=200, M=16)
    p.set_ef(10)
    p.set_num_threads(4)

    remaining = db_session.query(People.id, DefineImages.id, Timeline.embedding)\
        .filter(People.user_id == current_user['id'])\
        .filter(DefineImages.person_access_key == People.access_key)\
        .filter(Timeline.image_id == DefineImages.image_id)\
        .all()
    for imageI in remaining:
        embedding = imageI[2][2:-2]
        embedding = np.expand_dims(np.fromstring(embedding, dtype='float32', sep=','), axis=0)
        p.add_items(embedding, np.array([imageI[1]]))
    p.save_index("indexes/index_" + str(current_user['secret_key']) + ".bin")

    return web.json_response({"result": {'message': 'Ok'}}, status=200)


@login_app.login_required
async def add_class(current_user, request):
    """
    ---
    description: This end-point allow to add class.
    tags:
    - Class Adding
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
    """
    req = await request.json()
    if 'class_access_key' not in req or req['class_access_key'] == "":
        access_key = str(uuid.uuid4())
        new_class = Classes(name=req['class_name'], user_id=current_user['id'], access_key=access_key)
        db_session.add(new_class)
        db_session.commit()
    else:
        new_class = Classes.query.filter_by(access_key=req['class_access_key'], user_id=current_user['id']).first()
        if 'class_name' in req and req['class_name'] != "":
            new_class.name = req['class_name']
            db_session.commit()

    for ak in (req.get('teacher_access_keys') or []):
        person = People.query.filter_by(access_key=ak, user_id=current_user['id'], type_role="teacher").first()
        if person and not PeopleClasses.query.filter_by(person_access_key=person.access_key).first():
            db_session.add(PeopleClasses(class_access_key=new_class.access_key, person_access_key=person.access_key))
            db_session.commit()

    return web.json_response({"result": {'message': 'success'}}, status=200)


@login_app.login_required
async def delete_class(current_user, request):
    """
    ---
    description: This end-point allow to delete class.
    tags:
    - Class Deleting
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "403":
            description: Bạn không có quyền xóa
    """
    req = await request.json()
    target_class = Classes.query.filter_by(access_key=req['class_access_key'], user_id=current_user['id']).first()
    if not target_class:
        return web.json_response({"result": {'message': 'Bạn không có quyền xóa lớp này'}}, status=403)

    db_session.delete(target_class)
    db_session.execute(delete(PeopleClasses).where(PeopleClasses.class_access_key == req['class_access_key']))
    db_session.commit()

    return web.json_response({"result": {'message': 'success'}}, status=200)


async def check_pickup(request):
    """
    ---
    description: This end-point allow to check pickup between parent and children.
    tags:
    - Pickup Checking
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
        "400":
            description: Vui lòng truyền secret key
        "403":
            description: Secret key không hợp lệ
    """
    req = await request.json()
    if 'secret_key' not in req:
        return web.json_response({"result": {'message': 'Vui lòng truyền secret key'}}, status=400)

    user = Users.query.filter_by(secret_key=req['secret_key']).first()
    if not user:
        return web.json_response({"result": {'message': 'Secret key không hợp lệ'}}, status=403)

    if 'id1' not in req or 'id2' not in req or 'timeline_id1' not in req or 'timeline_id2' not in req:
        return web.json_response({"result": {'message': 'Vui lòng tryền đầy đủ dữ liệu'}}, status=400)

    person1 = None
    person2 = None
    if int(req['id1']) != -1:
        person1 = People.query.filter_by(access_key=req['id1'], user_id=user.id).first()
        if not person1:
            return web.json_response({"result": {'message': 'Bạn không có quyền thay đổi'}}, status=403)
    if int(req['id2']) != -1:
        person2 = People.query.filter_by(access_key=req['id2'], user_id=user.id).first()
        if not person2:
            return web.json_response({"result": {'message': 'Bạn không có quyền thay đổi'}}, status=403)

    if not person1 or person1.type_role == "parent" or person1.type_role == "teacher":
        pk = PickUp(child_access_key=int(req['id2']), picker_access_key=int(req['id1']),
                    child_timeline=int(req['timeline_id2']), parent_timeline=int(req['timeline_id1']))
    else:
        pk = PickUp(child_access_key=int(req['id1']), picker_access_key=int(req['id2']),
                    child_timeline=int(req['timeline_id1']), parent_timeline=int(req['timeline_id2']))
    db_session.add(pk)
    db_session.commit()

    return web.json_response({"result": {'message': 'success'}}, status=200)


@login_app.login_required
async def get_pickup(current_user, request):
    """
    ---
    description: This end-point allow to get all pickup between parent and children.
    tags:
    - Pickup Listing
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
    """
    page = int(request.match_info.get('page', '1'))
    if page <= 0:
        page = 1
    page_size = 10
    args = validate_request(request.rel_url.query, ['name'], {'name': ''})

    People_alias = aliased(People)
    Timeline_alias = aliased(Timeline)
    all_pickups_available = db_session.query(
        PickUp.id, Timeline.timestamp, People.name, People_alias.name, Timeline.image_id, Timeline_alias.image_id
    ).filter(
        Timeline.user_id == current_user['id'],
        Timeline_alias.user_id == current_user['id'],
        PickUp.child_access_key == People.access_key,
        PickUp.picker_access_key == People_alias.id,
        PickUp.child_timeline == Timeline.id,
        PickUp.picker_timeline == Timeline_alias.id,
        func.lower(People.name).like('%' + args['name'].lower() + '%') |
        func.lower(People_alias.name).like('%' + args['name'].lower() + '%')
    ).order_by(Timeline.timestamp.desc())

    all_pickups = all_pickups_available.limit(page_size).offset((page - 1) * page_size).all()
    safe_pickups = all_pickups_available.filter(
        PickUp.child_access_key == ChildrenPicker.child_access_key &
        PickUp.picker_access_key == ChildrenPicker.picker_access_key
    ).all()

    pickup_array = {}
    for u in all_pickups:
        pickup_array[str(u[0])] = {'timestamp': u[1], 'child_name': u[2], 'parent_name': u[3],
                                   'child_image_id': u[4], 'parent_image_id': u[5], 'is_acceptable': False}
    for u in safe_pickups:
        if str(u[0]) in pickup_array:
            pickup_array[str(u[0])]['is_acceptable'] = True

    return web.json_response({"result": {
        "number_of_pickup": len(all_pickups_available.all()),
        "pickup_list": [pickup_array[u] for u in pickup_array.keys()],
    }}, status=200)


@login_app.login_required
async def get_class(current_user, request):
    """
    ---
    description: This end-point allow to get all classes.
    tags:
    - Class Listing
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
    """
    page = int(request.match_info.get('page', '1'))
    if page <= 0:
        page = 1
    page_size = 10
    args = validate_request(request.rel_url.query,
        ['name', 'class_access_key'], {'name': '', 'class_access_key': '%%'})

    all_classes_count = db_session.query(Classes.access_key, Classes.name)\
        .filter(Classes.user_id == current_user['id'])\
        .filter(func.lower(Classes.name).like('%' + args['name'].lower() + '%') &
                Classes.access_key.like(args['class_access_key']))

    all_classes = db_session.query(Classes.access_key, Classes.name, func.count(Classes.access_key))\
        .filter(Classes.user_id == current_user['id'],
                Classes.access_key == PeopleClasses.class_access_key,
                People.access_key == PeopleClasses.person_access_key,
                People.type_role == 'student')\
        .group_by(Classes.access_key).all()

    all_classes_teacher = db_session.query(Classes.access_key, Classes.name, People.name, DefineImages.image_id)\
        .filter(Classes.user_id == current_user['id'],
                Classes.access_key == PeopleClasses.class_access_key,
                People.access_key == PeopleClasses.person_access_key,
                People.type_role == 'teacher',
                DefineImages.person_access_key == People.access_key)\
        .group_by(Classes.access_key).all()

    class_array = {}
    for u in all_classes_count.limit(page_size).offset((page - 1) * page_size).all():
        class_array[str(u[0])] = {'access_key': u[0], 'classname': u[1], 'number_of_student': 0}
    for u in all_classes:
        if str(u[0]) in class_array:
            class_array[str(u[0])]['number_of_student'] = u[2]
    for u in all_classes_teacher:
        if str(u[0]) in class_array:
            class_array[str(u[0])]['teachers'] = {'name': u[2], 'image_id': u[3]}

    return web.json_response({"result": {
        "number_of_class": len(all_classes_count.all()),
        "class_list": [class_array[u] for u in class_array.keys()],
    }}, status=200)


@login_app.login_required
async def people_list(current_user, request):
    """
    ---
    description: This end-point allow to get all people.
    tags:
    - People Listing
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
    """
    page = int(request.match_info.get('page', '1'))
    if page <= 0:
        page = 1
    page_size = 10
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
    args = validate_request(request.rel_url.query,
        ['name', 'type_role', 'class_name', 'access_key', 'begin', 'end'],
        {'name': "%{}%".format(''), 'type_role': "%{}%".format(''), 'access_key': "%{}%".format(''),
         'class_name': "%{}%".format(''), 'begin': today, 'end': today + 86400000})

    sub = db_session.query(DefineImages.image_id, DefineImages.id.label('id')).subquery()
    all_classes = Classes.query.filter_by(user_id=current_user['id']).all()

    all_people_count = db_session.query(
        DefineImages.person_access_key, DefineImages.image_id,
        People.name, People.access_key, People.type_role, People.age, People.gender, People.phone, People.description
    ).filter(
        People.user_id == current_user['id'],
        People.access_key == DefineImages.person_access_key,
        func.lower(People.name).like("%" + args['name'].lower() + "%") &
        People.type_role.like(args['type_role']) &
        People.access_key.like(args['access_key'])
    )

    if len(all_classes) > 0:
        all_people_count = all_people_count.filter(Classes.name.like(args['class_name']))

    all_people_count = all_people_count\
        .group_by(DefineImages.person_access_key, People.name, People.access_key)\
        .filter(DefineImages.id == sub.c.id)

    all_people_have_class = db_session.query(
        DefineImages.person_access_key, DefineImages.image_id,
        People.name, People.access_key, People.type_role, Classes.name, Classes.access_key
    ).filter(
        People.user_id == current_user['id'],
        People.access_key == DefineImages.person_access_key,
        People.access_key == PeopleClasses.person_access_key,
        Classes.access_key == PeopleClasses.class_access_key
    ).all()

    current_checkin = db_session.query(
        Timeline.person_access_key, func.min(Timeline.timestamp), func.max(Timeline.timestamp),
        Timeline.image_id, People.name
    ).filter(
        Timeline.user_id == current_user['id'],
        People.access_key == Timeline.person_access_key,
        Timeline.timestamp >= int(args['begin']),
        Timeline.timestamp <= int(args['end'])
    ).group_by(Timeline.person_access_key, People.name).all()

    all_people = all_people_count.limit(page_size).offset((page - 1) * page_size).all()

    people_array = {}
    for u in all_people:
        people_array[str(u[0])] = {
            'name': u[2], 'image_ids': u[1], 'begin': '--', 'end': '--',
            'checkin': False, 'access_key': u[3], 'type_role': u[4],
            'class_name': None, 'class_access_key': None,
            'age': u[5], 'gender': u[6], 'phone': u[7], 'description': u[8]
        }
    for u in (current_checkin or []):
        if str(u[0]) in people_array:
            people_array[str(u[0])].update({
                'name': u[4], 'image_ids': u[3],
                'begin': str(u[1]), 'end': str(u[2]), 'checkin': True,
            })
    for u in all_people_have_class:
        if str(u[0]) in people_array:
            people_array[str(u[0])]['class_name'] = u[5]
            people_array[str(u[0])]['class_access_key'] = u[6]

    return web.json_response({"result": {
        "people_list": [people_array[u] for u in people_array.keys()],
        'number_of_people': len(all_people_count.all()),
        'number_of_current_checkin': len(current_checkin),
    }}, status=200)


@login_app.login_required
async def data_a(current_user, request):
    """
    ---
    description: This end-point allow to get dashboard data.
    tags:
    - Data Profiling
    produces:
    - text/json
    responses:
        "200":
            description: successful operation
    """
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000

    all_people = db_session.query(
        DefineImages.person_access_key, DefineImages.image_id, People.name, People.access_key
    ).filter(
        People.user_id == current_user['id'],
        People.access_key == DefineImages.person_access_key
    ).group_by(DefineImages.person_access_key, People.name, People.access_key).all()

    current_checkin = db_session.query(
        Timeline.person_access_key, Timeline.timestamp, Timeline.image_id, Timeline.mask, Timeline.yaw, People.name
    ).filter(
        Timeline.user_id == current_user['id'],
        People.access_key == Timeline.person_access_key,
        Timeline.timestamp >= today
    ).group_by(Timeline.person_access_key, People.name).all()

    current_timeline = db_session.query(
        Timeline.person_access_key, Timeline.timestamp, Timeline.image_id, Timeline.mask, Timeline.yaw, People.name
    ).filter(
        Timeline.user_id == current_user['id'],
        People.access_key == Timeline.person_access_key
    ).order_by(Timeline.timestamp.desc()).limit(10).all()

    strangers = db_session.query(
        Timeline.person_access_key, Timeline.timestamp, Timeline.image_id, Timeline.mask, Timeline.yaw
    ).filter(
        Timeline.user_id == current_user['id'],
        Timeline.person_access_key == -1
    ).order_by(Timeline.timestamp.desc()).limit(10).all()

    t = r = a = 0
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)

    return web.json_response({"result": {
        'secret_key': current_user['secret_key'],
        'number_of_people': len(all_people),
        'number_of_current_checkin': len(current_checkin),
        'current_timeline': [{'name': u[5], 'image_id': u[2], 'timestamp': str(u[1]), 'mask': u[3], 'yaw': u[4]} for u in current_timeline],
        'strangers': [{'image_id': u[2], 'timestamp': str(u[1]), 'mask': u[3], 'yaw': u[4]} for u in strangers],
        'gpu': {'t': t, 'r': r, 'a': a}
    }}, status=200)


async def images(request):
    """Serve a stored face image."""
    secret_key = request.match_info.get('secret_key', '')
    image_id = request.match_info.get('image_id', '')
    return web.FileResponse('images/' + secret_key + "/" + image_id + '.jpg')


# ══════════════════════════════════════════════════════════════════════════════
# APP / SERVER
# ══════════════════════════════════════════════════════════════════════════════

app = web.Application(client_max_size=200 * 1024**2)
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))


def create_error_middleware(overrides):
    @web.middleware
    async def error_middleware(request, handler):
        try:
            Base.metadata.create_all(bind=engine)
            resp = await handler(request)
            db_session.remove()
            return resp
        except web.HTTPException as ex:
            override = overrides.get(ex.status)
            if override:
                resp = await override(request, ex)
                resp.set_status(ex.status)
                return resp
        except Exception as e:
            print(traceback.format_exc())
            resp = await overrides[500](request, web.HTTPError(text=traceback.format_exc()))
            resp.set_status(500)
            return resp
    return error_middleware


async def handle_403(request, ex):
    raise web.HTTPFound(location='./')

async def handle_404(request, ex):
    raise web.HTTPFound(location='./')

async def handle_500(request, ex):
    raise web.HTTPFound(location='./')


app.middlewares.append(create_error_middleware({
    403: handle_403,
    404: handle_404,
    500: handle_500,
}))

app.router.add_route('*',   '/',                             index,        name="index")
app.router.add_route('GET', '/client',                       client_a)
app.router.add_route('GET', '/logout',                       logout)
app.router.add_route('POST', '/facerec',                     facerec)
app.router.add_route('POST', '/facereg',                     facereg)
app.router.add_route('POST', '/delete_image',                delete_image)
app.router.add_route('POST', '/add_class',                   add_class)
app.router.add_route('POST', '/delete_class',                delete_class)
app.router.add_route('POST', '/check_pickup',                check_pickup)
app.router.add_route('GET',  '/pickup_list/{page}',          get_pickup)
app.router.add_route('GET',  '/class_list/{page}',           get_class)
app.router.add_route('GET',  '/people_list/{page}',          people_list)
app.router.add_route('GET',  '/data',                        data_a)
app.router.add_route('GET',  '/images/{secret_key}/{image_id}', images)
app.add_routes([web.static('/static', 'static')])

setup_swagger(app, swagger_url="./api/v1/doc", ui_version=3)

if __name__ == "__main__":
    web.run_app(app, port=5002)
