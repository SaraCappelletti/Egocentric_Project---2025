import json
import os

def calculate_iou(start1, end1, start2, end2):
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0

def parse_ground_truth(gt_json):
    # Crea una mappa: (clip_uid, annotation_uid, query_idx) → (query, start, end)
    gt_map = {}
    for video in gt_json.get("videos", []):
        for clip in video.get("clips", []):
            clip_uid = clip.get("clip_uid")
            for ann in clip.get("annotations", []):
                annotation_uid = ann.get("annotation_uid")
                for query_idx, query in enumerate(ann.get("language_queries", [])):
                    if not all(k in query for k in ("query", "video_start_sec", "video_end_sec")):
                        continue  # Skippa query malformate
                    key = (clip_uid, annotation_uid, query_idx)
                    gt_map[key] = {
                        "query": query["query"].strip().lower(),
                        "gt_start": query["video_start_sec"],
                        "gt_end": query["video_end_sec"]
                    }
    return gt_map


def select_top_queries(predictions_path, ground_truth_path, output_path, top_k=50):
    with open(predictions_path, "r") as f:
        predictions = json.load(f)["results"]

    with open(ground_truth_path, "r") as f:
        gt_json = json.load(f)
        gt_map = parse_ground_truth(gt_json)

    scored = []

    for pred in predictions:
        key = (pred["clip_uid"], pred["annotation_uid"], pred["query_idx"])
        if key not in gt_map:
            continue

        pred_times = pred.get("predicted_times", [])
        if not pred_times:
            continue

        pred_start, pred_end = pred_times[0]  # usa la top-1 prediction
        gt = gt_map[key]
        iou = calculate_iou(pred_start, pred_end, gt["gt_start"], gt["gt_end"])

        scored.append({
            "query": gt["query"],
            "clip_uid": pred["clip_uid"],
            "annotation_uid": pred["annotation_uid"],
            "query_idx": pred["query_idx"],
            "clip_start_sec": pred_start,
            "clip_end_sec": pred_end,
            "iou": iou
        })

    scored.sort(key=lambda x: x["iou"], reverse=True)
    selected = scored[:top_k]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(selected, f, indent=4)

    print(f"✅ Selected top {top_k} queries saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--ground_truth_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    select_top_queries(
        predictions_path=args.predictions_path,
        ground_truth_path=args.ground_truth_path,
        output_path=args.output_path,
        top_k=args.top_k
    )
