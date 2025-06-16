import cv2
import numpy as np
import json
import os
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2
from app.services.model_predictor import predict_from_keypoints_df
from app.services.image_visualizer import generate_pose_visualization
from app.services.pose_estimation import get_keypoints

def run_openpose(image, opWrapper):
    datum = op.Datum()
    datum.cvInputData = image
    datums = op.VectorDatum()
    datums.append(datum)
    opWrapper.emplaceAndPop(datums)
    return datum

def get_coords(keypoints, index):
    try:
        x, y, c = keypoints[0][index]
        return (int(x), int(y)) if c > 0.1 else None
    except:
        return None

def detect_facing_direction(keypoints):
    score = 0
    nose = get_coords(keypoints, 0)
    left_shoulder = get_coords(keypoints, 5)
    left_elbow = get_coords(keypoints, 6)
    left_wrist = get_coords(keypoints, 7)
    left_knee = get_coords(keypoints, 13)

    if nose and left_shoulder:
        score += 1 if nose[0] < left_shoulder[0] else -1
    if left_shoulder and left_wrist:
        score += 1 if left_shoulder[0] > left_wrist[0] else -1
    if left_shoulder and left_knee:
        score += 1 if left_shoulder[0] > left_knee[0] else -1
    if left_shoulder and left_elbow:
        score += 1 if left_shoulder[0] > left_elbow[0] else -1

    return score

def process_pose_from_bytes(image_bytes):
    params = {
        "model_pose": "BODY_25",
        "hand": True,
        "number_people_max": 1,
        # "net_resolution": "-1x160",
        "model_folder": "/root/openpose/models"
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image = bytes_to_cv2(image_bytes)
    datum = run_openpose(image, opWrapper)
    keypoints = datum.poseKeypoints

    processed_image = image_bytes
    processed_keypoints = keypoints
    is_flipped = False

    if keypoints is not None:
        direction_score = detect_facing_direction(keypoints)
        if direction_score > 0:
            print("Flip image ke kanan")
            flipped = cv2.flip(image, 1)
            datum = run_openpose(flipped, opWrapper)
            keypoints = datum.poseKeypoints
            is_flipped = True

    hand_kpts = datum.handKeypoints[1] if datum.handKeypoints is not None else None  # right hand

    keypoint_df = get_keypoints(keypoints, hand_kpts)
    print(keypoint_df)

    # Prediksi model
    hasil_prediksi = predict_from_keypoints_df(keypoint_df)

    # # Generate visualisasi dan ambil path gambar
    # gambar_path = generate_pose_visualization(processed_image, processed_keypoints, hasil_prediksi, is_flipped)

    # # Tambahkan ke hasil prediksi
    # hasil_prediksi["gambar_path"] = gambar_path

    return hasil_prediksi