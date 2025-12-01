from flask import Blueprint, request, jsonify, send_from_directory, render_template, send_file
from app.services.pose_estimation import process_pose_from_bytes
from app.services.video_processor import process_video
from app.services.job_manager import create_job, get_job
from app.services.tts_service import process_tts, get_supported_languages, AUDIO_OUTPUT_DIR
from datetime import datetime, timedelta
from app.ws_handlers import summary_storage
import os
import threading

pose_bp = Blueprint('pose', __name__)

@pose_bp.route('/predict/image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        result = process_pose_from_bytes(image_bytes)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@pose_bp.route('/output_images/<path:filename>')
def serve_output_image(filename):
    directory = os.path.join(os.getcwd(), 'output_images')
    return send_from_directory(directory, filename, mimetype='image/png')

@pose_bp.route("/predict/video", methods=["POST"])
def predict_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    interval = request.form.get('interval', default=30, type=int)

    job_id = create_job()

    job_folder = os.path.join("temp_jobs", job_id)
    os.makedirs(job_folder, exist_ok=True)

    video_path = os.path.join(job_folder, "video.mp4")
    file.save(video_path) 

    # Menggunakan threading agar tidak memblokir server
    threading.Thread(target=process_video, args=(job_folder, job_id, interval)).start()

    return jsonify({"job_id": job_id})

@pose_bp.route("/predict/video/result", methods=["GET"])
def get_video_result():
    job_id = request.args.get("job_id")

    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job)

@pose_bp.route("/websocket/summary", methods=["GET"])
def get_summary():
    sid = request.args.get("sid")

    if sid not in summary_storage:
        return jsonify({"error": "Summary tidak ditemukan"}), 404

    summary_entry = summary_storage[sid]
    timestamp = summary_entry["timestamp"]

    if datetime.now() - timestamp > timedelta(hours=24):
        del summary_storage[sid]
        return jsonify({"error": "Summary expired"}), 410

    return jsonify(summary_entry["data"])

@pose_bp.route("/ws-client")
def serve_client():
    return render_template("client.html")

@pose_bp.route("/client")
def serve_client_predict():
    return render_template("predict.html")

# === ROUTE BARU KHUSUS FITUR TTS (PASTIKAN BAGIAN INI ADA) ===

@pose_bp.route("/tts-simulation")
def tts_simulation_page():
    """Halaman dummy untuk mencoba fitur TTS"""
    languages = get_supported_languages()
    return render_template("tts_dummy.html", languages=languages)

@pose_bp.route("/api/generate-tts", methods=["POST"])
def generate_tts_api():
    """API untuk memproses teks -> translate -> audio"""
    data = request.json
    text = data.get('text')
    lang = data.get('lang', 'id')
    
    result = process_tts(text, lang)
    
    if "error" in result:
        return jsonify(result), 500
        
    return jsonify(result)

@pose_bp.route('/audio/<path:filename>')
def serve_audio(filename):
    """Untuk menyajikan file audio hasil generate"""
    return send_from_directory(AUDIO_OUTPUT_DIR, filename)