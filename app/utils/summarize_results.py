from collections import defaultdict, Counter

def summarize_results(results):
    count = len(results)

    skor_counter = defaultdict(Counter)
    skor_total = defaultdict(int)
    skor_min = defaultdict(lambda: float('inf'))
    skor_max = defaultdict(lambda: float('-inf'))

    sudut_data = defaultdict(float)
    sudut_count = defaultdict(int)

    feedback_counter = Counter()
    rula_summary_counter = Counter()
    reba_summary_counter = Counter()

    for res in results:
        # Hitung skor RULA/REBA
        for key, value in res.items():
            if key.endswith("_score"):
                skor_counter[key][value] += 1
                skor_total[key] += value
                skor_min[key] = min(skor_min[key], value)
                skor_max[key] = max(skor_max[key], value)

        # Hitung sudut
        for sudut_key, sudut_val in res.get("details", {}).items():
            if sudut_key.startswith("sudut_"):
                sudut_data[sudut_key] += sudut_val
                sudut_count[sudut_key] += 1

        # Hitung feedback
        feedback_text = res.get("feedback", "")
        if feedback_text:
            feedback_counter[feedback_text] += 1

        # Hitung summary
        summary_data = res.get("summary", {})
        if "rula_summary" in summary_data:
            rula_summary_counter[summary_data["rula_summary"]] += 1
        if "reba_summary" in summary_data:
            reba_summary_counter[summary_data["reba_summary"]] += 1

    # Hitung statistik skor
    majority_scores = {k: v.most_common(1)[0][0] for k, v in skor_counter.items()}
    average_scores = {k: skor_total[k] / count for k in skor_counter}
    min_scores = {k: skor_min[k] for k in skor_counter}
    max_scores = {k: skor_max[k] for k in skor_counter}

    # Hitung rata-rata sudut
    average_sudut = {k: sudut_data[k] / sudut_count[k] for k in sudut_data}

    # Feedback dan summary paling umum
    feedback_summary = feedback_counter.most_common(1)[0][0] if feedback_counter else "-"
    rula_summary = rula_summary_counter.most_common(1)[0][0] if rula_summary_counter else "-"
    reba_summary = reba_summary_counter.most_common(1)[0][0] if reba_summary_counter else "-"

    def generate_dynamic_summary(count, majority_scores, average_scores, min_scores, max_scores, average_sudut, rula_summary, reba_summary, feedback_summary):
        rula_avg = average_scores.get("rula_final_score", 0)
        reba_avg = average_scores.get("reba_final_score", 0)
        rula_max = max_scores.get("rula_final_score", 0)
        reba_max = max_scores.get("reba_final_score", 0)
        rula_major = majority_scores.get("rula_final_score", 0)
        reba_major = majority_scores.get("reba_final_score", 0)

        def interpret_score(avg, major, maks, label):
            if major >= 5 or maks >= 7:
                return f"{label} menunjukkan postur yang sangat berisiko. Terdapat frame dengan skor tinggi ({maks}), mayoritas juga di level tinggi ({major})."
            elif major >= 3 or avg >= 3:
                return f"{label} menunjukkan postur yang perlu perhatian. Rata-rata skor berada di sekitar {avg:.2f}, mayoritas {major}, dan skor maksimum {maks}."
            else:
                return f"{label} menunjukkan postur relatif aman. Rata-rata {avg:.2f}, mayoritas {major}, dan skor maksimum hanya {maks}."

        summary_feedback = (
            f"Analisa dari {count} frame:\n\n"
            f"{interpret_score(rula_avg, rula_major, rula_max, 'RULA')}\n"
            f"{interpret_score(reba_avg, reba_major, reba_max, 'REBA')}\n\n"
            f"Ringkasan teks terbanyak:\n {feedback_summary}\n\n"
            # f"Rata-rata sudut tubuh:\n" +
            # "\n".join([f"- {k.replace('_', ' ').capitalize()}: {v:.2f}°" for k, v in average_sudut.items()]) + "\n"
        )

        return summary_feedback

    summary_feedback = generate_dynamic_summary(
        count,
        majority_scores,
        average_scores,
        min_scores,
        max_scores,
        average_sudut,
        rula_summary,
        reba_summary,
        feedback_summary
    )

    return {
        "summary": summary_feedback,
        "majority_scores": majority_scores,
        "average_scores": average_scores,
        "min_scores": min_scores,
        "max_scores": max_scores,
        "average_sudut": average_sudut,
        "total_frames": count,
        "most_common_feedback": feedback_summary,
        "rula_summary": rula_summary,
        "reba_summary": reba_summary,
        "representative_image": None
    }