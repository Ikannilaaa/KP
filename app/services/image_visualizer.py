import os
import cv2
import numpy as np
from datetime import datetime

def get_color_by_score(score):
    if score <= 1:
        return (0, 255, 0)      # Green
    elif score == 2:
        return (173, 255, 47)   # Yellow-green
    elif score == 3:
        return (0, 255, 255)    # Yellow
    elif score == 4:
        return (0, 165, 255)    # Orange
    elif score == 5:
        return (0, 140, 255)    # Darker Orange
    elif score == 6:
        return (0, 69, 255)     # Orange-red
    else:  # score >= 7
        return (0, 0, 255)      # Red

def get_risk_label(score):
    if score <= 1:
        return "Negligible"
    elif score <= 3:
        return "Low"
    elif score <= 7:
        return "Medium"
    elif score <= 10:
        return "High"
    else:
        return "Very High"

def generate_pose_visualization(image_bytes, keypoints, hasil_prediksi, is_flipped):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    scale_factor = height / 720  # Sesuaikan referensi jika mau

    blank_space_width = int(180 * scale_factor)
    blank = np.ones((height, blank_space_width, 3), dtype=np.uint8) * 255
    img = np.hstack((blank, img))

    # --- Penyesuaian ukuran
    radius = int(25 * scale_factor)
    font_scale = 0.5 * scale_factor
    font_thickness = max(1, int(1 * scale_factor))
    spacing = int(20 * scale_factor)
    badge_w = int(150 * scale_factor)
    badge_h = int(40 * scale_factor)
    split_radius = int(15 * scale_factor)
    legend_x = int(15 * scale_factor)
    text_offset_y = int(30 * scale_factor)

    mapping_rula = {
        "sudut_lutut": hasil_prediksi.get("rula_leg_score", 0),
        "sudut_siku": hasil_prediksi.get("rula_lower_arm_score", 0),
        "sudut_leher": hasil_prediksi.get("rula_neck_score", 0),
        "sudut_punggung": hasil_prediksi.get("rula_trunk_score", 0),
        "sudut_pergelangan": hasil_prediksi.get("rula_wrist_score", 0),
        "sudut_bahu": hasil_prediksi.get("rula_upper_arm_score", 0),
    }

    mapping_reba = {
        "sudut_lutut": hasil_prediksi.get("reba_leg_score", 0),
        "sudut_siku": hasil_prediksi.get("reba_lower_arm_score", 0),
        "sudut_leher": hasil_prediksi.get("reba_neck_score", 0),
        "sudut_punggung": hasil_prediksi.get("reba_trunk_score", 0),
        "sudut_pergelangan": hasil_prediksi.get("reba_wrist_score", 0),
        "sudut_bahu": hasil_prediksi.get("reba_upper_arm_score", 0),
    }

    point_mapping = {
        "sudut_lutut": 13 if is_flipped else 10,
        "sudut_siku": 6 if is_flipped else 3,
        "sudut_leher": 18 if is_flipped else 17,
        "sudut_punggung": 12 if is_flipped else 9,
        "sudut_pergelangan": 7 if is_flipped else 4,
        "sudut_bahu": 5 if is_flipped else 2,
    }

    for key, index in point_mapping.items():
        try:
            x, y, conf = keypoints[0][index]
            x += blank_space_width
            if conf > 0.1:
                overlay = img.copy()
                rula_score = mapping_rula[key]
                reba_score = mapping_reba[key]
                rula_color = get_color_by_score(rula_score)
                reba_color = get_color_by_score(reba_score)
                center = (int(x), int(y))
                axes = (radius, radius)
                cv2.ellipse(overlay, center, axes, 0, 90, 270, rula_color, -1)
                cv2.ellipse(overlay, center, axes, 0, -90, 90, reba_color, -1)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

                sudut_val = hasil_prediksi.get("details", {}).get(key, None)
                if sudut_val is not None:
                    label = f"{sudut_val:.0f}d"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    text_x = int(x) - w // 2 - 5
                    text_y = int(y) - text_offset_y
                    cv2.putText(img, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness + 3)
                    cv2.putText(img, label, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        except:
            continue

    # Final score
    rula_final = hasil_prediksi.get("rula_final_score", 0)
    reba_final = hasil_prediksi.get("reba_final_score", 0)
    rula_label = get_risk_label(rula_final)
    reba_label = get_risk_label(reba_final)

    def label_color(label):
        return {
            "Negligible": (144, 238, 144),
            "Low": (0, 255, 0),
            "Medium": (0, 255, 255),
            "High": (0, 165, 255),
            "Very High": (0, 0, 255)
        }.get(label, (255, 255, 255))

    # Badge
    badge_x, badge_y = int(10 * scale_factor), int(10 * scale_factor)
    cv2.rectangle(img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (50, 50, 50), -1)
    cv2.putText(img, f"RULA: {rula_final}", (badge_x + int(10 * scale_factor), badge_y + int(15 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color(rula_label), font_thickness)
    cv2.putText(img, f"REBA: {reba_final}", (badge_x + int(10 * scale_factor), badge_y + int(30 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color(reba_label), font_thickness)

    # Legend
    rula_labels = [(1, "1: Acceptable"), (2, "2: Acceptable"), (3, "3: Investigate"),
                   (4, "4: Investigate"), (5, "5: Change Soon"), (6, "6: Change Soon"), (7, "7: Implement Change")]

    reba_labels = [(1, "1: Negligible"), (2, "2: Low"), (3, "3: Low"),
                   (4, "4: Medium"), (5, "5: Medium"), (6, "6: Medium"),
                   (7, "7: Medium"), (8, "8+: High")]

    rula_block_height = spacing * len(rula_labels) + int(25 * scale_factor)
    reba_block_height = spacing * len(reba_labels) + int(25 * scale_factor)
    total_legend_height = rula_block_height + reba_block_height + 10
    legend_y_start = height - total_legend_height

    # Background blocks
    cv2.rectangle(img, (legend_x - 5, legend_y_start - 5),
                  (legend_x + int(150 * scale_factor), legend_y_start + rula_block_height), (30, 30, 30), -1)
    cv2.rectangle(img, (legend_x - 5, legend_y_start + rula_block_height + 5),
                  (legend_x + int(150 * scale_factor), legend_y_start + rula_block_height + 5 + reba_block_height), (30, 30, 30), -1)

    # Split legend
    split_example_center = (legend_x + int(20 * scale_factor), legend_y_start - int(30 * scale_factor))
    split_axes = (split_radius, split_radius)
    rula_example_color = get_color_by_score(3)
    reba_example_color = get_color_by_score(2)
    cv2.ellipse(img, split_example_center, split_axes, 0, 90, 270, rula_example_color, -1)
    cv2.ellipse(img, split_example_center, split_axes, 0, 270, 450, reba_example_color, -1)
    cv2.ellipse(img, split_example_center, split_axes, 0, 0, 360, (0, 0, 0), 1)

    cv2.putText(img, "Kiri: RULA", (split_example_center[0] + int(20 * scale_factor), split_example_center[1] - int(3 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale_factor, (0, 0, 0), font_thickness)
    cv2.putText(img, "Kanan: REBA", (split_example_center[0] + int(20 * scale_factor), split_example_center[1] + int(13 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale_factor, (0, 0, 0), font_thickness)

    # Judul + isi legend RULA
    cv2.putText(img, "RULA", (legend_x, legend_y_start + int(15 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_factor, (255, 255, 255), font_thickness)
    for i, (score, desc) in enumerate(rula_labels):
        cy = legend_y_start + int(25 * scale_factor) + i * spacing
        color = get_color_by_score(score)
        cv2.circle(img, (legend_x + radius, cy), radius // 4, color, -1)
        cv2.putText(img, desc, (legend_x + 2 * radius + 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale_factor, (255, 255, 255), font_thickness)

    # Judul + isi legend REBA
    reba_y_start = legend_y_start + rula_block_height + int(20 * scale_factor)
    cv2.putText(img, "REBA", (legend_x, reba_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale_factor, (255, 255, 255), font_thickness)
    for i, (score, desc) in enumerate(reba_labels):
        cy = reba_y_start + int(10 * scale_factor) + i * spacing
        color = get_color_by_score(score)
        cv2.circle(img, (legend_x + radius, cy), radius // 4, color, -1)
        cv2.putText(img, desc, (legend_x + 2 * radius + 6, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale_factor, (255, 255, 255), font_thickness)

    # Save image
    folder_path = os.path.join("output_images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    filename = datetime.now().strftime("%H%M%S_%f") + "_hasil.png"
    filepath = os.path.join(folder_path, filename)
    link_image = "https://vps.danar.site/model1/" + filepath
    cv2.imwrite(filepath, img)

    return link_image
