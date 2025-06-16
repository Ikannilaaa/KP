import cv2
import numpy as np
import json
import os
import subprocess
import pandas as pd
from openpose import pyopenpose as op
from app.utils.image_converter import bytes_to_cv2
from app.services.model_predictor import predict_from_keypoints_df
from app.services.image_visualizer import generate_pose_visualization

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
        "model_folder": "/root/openpose/models"
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    image = bytes_to_cv2(image_bytes)
    datum = run_openpose(image, opWrapper)
    keypoints = datum.poseKeypoints

    output_image = datum.cvOutputData
    _, encoded_img = cv2.imencode(".jpg", output_image)
    processed_image = encoded_img.tobytes()

    # processed_image = image_bytes
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

    # Prediksi model
    hasil_prediksi = predict_from_keypoints_df(keypoint_df)

    # Generate visualisasi dan ambil path gambar
    gambar_path = generate_pose_visualization(processed_image, processed_keypoints, hasil_prediksi, is_flipped)

    # Tambahkan ke hasil prediksi
    hasil_prediksi["gambar_path"] = gambar_path

    return hasil_prediksi

def run_openpose_on_folder(image_folder, output_json_folder, output_image_folder):
    openpose_bin_path = "/root/openpose/build/examples/openpose/openpose.bin"
    model_folder = "/root/openpose/models"

    command = [
        openpose_bin_path,
        "--image_dir", image_folder,
        "--write_json", output_json_folder,
        "--write_images", output_image_folder,
        "--model_folder", model_folder,
        "--hand", "True",
        "--display", "0",
        "--number_people_max", "1",
        "--disable_multi_thread"
    ]

    subprocess.run(command, check=True)

def process_openpose_results(json_folder, image_folder):
    results = []

    for file in sorted(os.listdir(json_folder)):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file)) as f:
                data = json.load(f)

            keypoints = None
            hand_right_keypoints = None

            if data['people']:
                person = data['people'][0]
                if "pose_keypoints_2d" in person and person["pose_keypoints_2d"]:
                    keypoints = np.array(person["pose_keypoints_2d"]).reshape(-1, 3)

                if "hand_right_keypoints_2d" in person and person["hand_right_keypoints_2d"]:
                    hand_right_keypoints = np.array(person["hand_right_keypoints_2d"]).reshape(-1, 3)

            if keypoints is not None:
                keypoint_df = get_keypoints_video(keypoints, hand_right_keypoints)

                hasil_prediksi = predict_from_keypoints_df(keypoint_df)

                # Path gambar input
                image_file = file.split('_keypoints')[0] + '.jpg'
                hasil_prediksi['gambar_path'] = os.path.join(image_folder, image_file)

                results.append(hasil_prediksi)

    return results

def get_keypoints(keypoints, hand_kpts):
    def get_point(index, keypoints_array):
        try:
            if keypoints_array is None:
                return -1.0, -1.0, -1.0
            x, y, c = keypoints_array[0][index]
            if any(np.isnan([x, y, c])) or c < 0.01:
                return -1.0, -1.0, -1.0
            return float(x), float(y), float(c)
        except:
            return -1.0, -1.0, -1.0
    
    keypoints_dict = {
        'hip': get_point(9, keypoints),
        'knee': get_point(10, keypoints),
        'ankle': get_point(11, keypoints),
        'shoulder': get_point(2, keypoints),
        'shoulder_left': get_point(5, keypoints),
        'elbow': get_point(3, keypoints),
        'elbow_left': get_point(6, keypoints),
        'wrist': get_point(4, keypoints),
        'hand_middle': get_point(12, hand_kpts) if hand_kpts is not None else (-1.0, -1.0, -1.0),
        'back': get_point(8, keypoints),
        'neck': get_point(1, keypoints),
        'head': get_point(17, keypoints),
        'nose': get_point(0, keypoints)
    }

    return keypoints_dict

def get_keypoints_video(keypoints, hand_kpts):
    def get_point(index, keypoints_array):
        try:
            if keypoints_array is None:
                return -1.0, -1.0, -1.0
            x, y, c = keypoints_array[index]
            if any(np.isnan([x, y, c])) or c < 0.01:
                return -1.0, -1.0, -1.0
            return float(x), float(y), float(c)
        except:
            return -1.0, -1.0, -1.0
    
    keypoints_dict = {
        'hip': get_point(9, keypoints),
        'knee': get_point(10, keypoints),
        'ankle': get_point(11, keypoints),
        'shoulder': get_point(2, keypoints),
        'shoulder_left': get_point(5, keypoints),
        'elbow': get_point(3, keypoints),
        'elbow_left': get_point(6, keypoints),
        'wrist': get_point(4, keypoints),
        'hand_middle': get_point(12, hand_kpts) if hand_kpts is not None else (-1.0, -1.0, -1.0),
        'back': get_point(8, keypoints),
        'neck': get_point(1, keypoints),
        'head': get_point(17, keypoints),
        'nose': get_point(0, keypoints)
    }

    return keypoints_dict

def check_video_direction(video_path, check_frame=10):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    direction_score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == check_frame:
            # Deteksi arah
            params = {
                "model_pose": "BODY_25",
                "hand": True,
                "number_people_max": 1,
                "model_folder": "/root/openpose/models"
            }
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()

            datum = run_openpose(frame, opWrapper)
            keypoints = datum.poseKeypoints
            if keypoints is not None:
                direction_score = detect_facing_direction(keypoints)

            opWrapper.stop()
            break

        frame_idx += 1

    cap.release()
    return direction_score

def flip_video(video_path):
    cap = cv2.VideoCapture(video_path)

    flipped_path = video_path.replace(".mp4", "_flipped.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        flipped = cv2.flip(frame, 1)

        if out is None:
            height, width = flipped.shape[:2]
            out = cv2.VideoWriter(flipped_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        out.write(flipped)

    cap.release()
    out.release()

    return flipped_path
