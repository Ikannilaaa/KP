import cv2
import os
import shutil
import json
import numpy as np
from app.services.pose_estimation import check_video_direction, flip_video, run_openpose_on_folder, process_openpose_results
from app.services.job_manager import update_job
from app.utils.summarize_results import summarize_results
from app.services.image_visualizer import generate_pose_visualization

def process_video(job_folder, job_id, interval):
    video_path = os.path.join(job_folder, "video.mp4")

    frames_folder = os.path.join(job_folder, "frames")
    json_folder = os.path.join(job_folder, "json")
    output_image_folder = os.path.join(job_folder, "rendered")

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    try:
        direction_score = check_video_direction(video_path)

        if direction_score > 0:
            video_path = flip_video(video_path)

        sample_video_to_folder(video_path, frames_folder, interval)
        run_openpose_on_folder(frames_folder, json_folder, output_image_folder)

        sampled_results = process_openpose_results(json_folder, frames_folder)
        final_result = summarize_results(sampled_results)

        if sampled_results:
            ranked = sorted(sampled_results, key=lambda x: max(x.get("reba_final_score", 0), x.get("rula_final_score", 0)), reverse=True)
            best_frame = ranked[0]

            keypoints_path = best_frame.get("gambar_path", "").replace("frames/", "json/").replace(".jpg", "_keypoints.json")
            original_jpg = os.path.basename(best_frame["gambar_path"])
            rendered_name = original_jpg.replace(".jpg", "_rendered.png")
            image_path = os.path.join(output_image_folder, rendered_name)

            if os.path.exists(image_path) and os.path.exists(keypoints_path):
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()

                with open(keypoints_path, 'r') as f:
                    keypoints_data = json.load(f)
                    keypoints_raw = keypoints_data["people"][0]["pose_keypoints_2d"]
                    processed_keypoints = np.array(keypoints_raw).reshape((-1, 3)).tolist()
                    if isinstance(processed_keypoints[0], list):  # jadi 2D
                        processed_keypoints = [processed_keypoints]

                output_image_path = generate_pose_visualization(
                    image_bytes, processed_keypoints, best_frame, is_flipped=(direction_score > 0)
                )

                final_result["representative_image"] = output_image_path

        update_job(job_id, final_result)

    finally:
        shutil.rmtree(job_folder)
        print(f"[DEBUG] Folder job {job_folder} sudah dibersihkan.")

def sample_video_to_folder(video_path, output_folder, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{frame_count:06d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)

        frame_count += 1

    cap.release()
