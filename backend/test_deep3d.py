"""
Test the Deep3D mask-removal pipeline.

Input:  ./deep3d/masked-face.jpg
Output: ./deep3d/unmasked-face.jpg

Usage:
    python test_deep3d.py
"""

import os
import numpy as np
import cv2
import torch
import face_alignment
from PIL import Image

from deep3d.util.load_mats import load_lm3d
from deep3d.options.test_options import TestOptions
from deep3d.models import create_model
from deep3d.util.visualizer import MyVisualizer
from deep3d.util.preprocess import align_img

INPUT_PATH  = './deep3d/masked-face.jpg'
OUTPUT_PATH = './deep3d/unmasked-face.jpg'

# ── Setup ──────────────────────────────────────────────────────────────────────

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

print('Loading face alignment model ...')
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', face_detector='blazeface')

print('Loading Deep3D model ...')
opt = TestOptions().parse()
model_deep3d = create_model(opt)
model_deep3d.setup(opt)
model_deep3d.device = device
model_deep3d.parallelize()
model_deep3d.eval()

visualizer = MyVisualizer(opt)
lm3d_std = load_lm3d('./deep3d/BFM')

# ── Functions (mirrors app.py) ─────────────────────────────────────────────────

def detect_landmark(img):
    """Detect 5 facial landmarks from a PIL image."""
    preds = fa.get_landmarks_from_image(np.array(img))
    if preds is None:
        raise RuntimeError('No face detected in the image.')
    preds = preds[0]
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
    """Render the reconstructed face and return as RGB numpy array."""
    visuals = model_deep3d.get_current_visuals()
    return visualizer.save_img(visuals)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not os.path.isfile(INPUT_PATH):
        print(f'Input not found: {INPUT_PATH}')
        raise SystemExit(1)

    input_img = Image.open(INPUT_PATH).convert('RGB')
    print(f'Input: {INPUT_PATH}  size={input_img.size}')

    print('Detecting landmarks ...')
    lm = detect_landmark(input_img)

    print('Running Deep3D reconstruction ...')
    reconstruct(input_img, lm)
    output_rgb = rasterize()

    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, output_bgr)
    print(f'Saved: {OUTPUT_PATH}')
