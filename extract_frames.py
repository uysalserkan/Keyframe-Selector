import shutil
import cv2
import os
import logging
from argparse import ArgumentParser

import torch
import clip
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, HDBSCAN

# ... (rest of imports and config)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)


def extract_frames(video_path, out_dir, fps=1):
    logger.info(f"Starting frame extraction from {video_path}")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    logger.info(f"Video FPS: {video_fps}, sampling at {fps} FPS (interval: {frame_interval} frames)")

    idx, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        idx += 1
    cap.release()
    logger.info(f"Extracted {saved} frames to {out_dir}")
    
    
def extract_clip_features(frame_dir):
    logger.info(f"Extracting CLIP features for frames in {frame_dir}")
    features = []
    frame_paths = sorted(os.listdir(frame_dir))
    
    if not frame_paths:
        logger.warning(f"No frames found in {frame_dir}")
        return np.array([]), []

    with torch.no_grad():
        for i, f in enumerate(frame_paths):
            if (i + 1) % 10 == 0 or i == 0:
                logger.debug(f"Processing frame {i+1}/{len(frame_paths)}")
            img = Image.open(os.path.join(frame_dir, f)).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(img_input)
            feat /= feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())

    logger.info(f"Successfully extracted features for {len(features)} frames")
    return np.vstack(features), frame_paths


def select_keyframes(features, frame_paths, num_keyframes=5, method="kmeans"):
    if method == "kmeans":
        if len(features) < num_keyframes:
            logger.warning(f"Requested {num_keyframes} keyframes but only have {len(features)} frames. Adjusting num_keyframes.")
            num_keyframes = len(features)

        logger.info(f"Selecting {num_keyframes} keyframes using KMeans clustering")
        kmeans = KMeans(n_clusters=num_keyframes, random_state=42)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        keyframes = []
        for i in range(num_keyframes):
            cluster_idxs = np.where(labels == i)[0]
            if len(cluster_idxs) == 0:
                continue
            cluster_feats = features[cluster_idxs]
            center = centers[i]
            distances = np.linalg.norm(cluster_feats - center, axis=1)
            best_idx = cluster_idxs[np.argmin(distances)]
            keyframes.append(frame_paths[best_idx])
            
    elif method == "hdbscan":
        logger.info("Selecting keyframes using HDBSCAN (density-based clustering)")
        # min_cluster_size=2 is a reasonable default for video frames
        # to ensure we at least try to group similar frames
        hdb = HDBSCAN(min_cluster_size=2)
        labels = hdb.fit_predict(features)
        
        unique_labels = np.unique(labels)
        keyframes = []
        
        # Labels -1 are noise points in HDBSCAN
        for label in unique_labels:
            cluster_idxs = np.where(labels == label)[0]
            cluster_feats = features[cluster_idxs]
            
            # Find the most representative point (medoid)
            # For noise (-1), we might still want to pick frames if they are distinct
            # or skip them. Here we pick one representative per cluster including noise.
            center = np.mean(cluster_feats, axis=0)
            distances = np.linalg.norm(cluster_feats - center, axis=1)
            best_idx = cluster_idxs[np.argmin(distances)]
            keyframes.append(frame_paths[best_idx])
            
        logger.info(f"HDBSCAN found {len(unique_labels)} clusters (including noise)")
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Keyframe selection complete. Selected {len(keyframes)} frames.")
    return keyframes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--frame_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--num_keyframes", type=int, default=6)
    parser.add_argument("--method", type=str, choices=["kmeans", "hdbscan"], default="kmeans",
                        help="Clustering method for keyframe selection")
    args = parser.parse_args()
    
    logger.info(f"Starting keyframe selection pipeline using {args.method}")
    extract_frames(args.video_path, args.frame_dir, fps=args.fps)
    features, frame_paths = extract_clip_features(args.frame_dir)
    
    if len(features) > 0:
        keyframes = select_keyframes(features, frame_paths, num_keyframes=args.num_keyframes, method=args.method)
        logger.info(f"Selected {len(keyframes)} keyframes")
            
    # Also save the keyframes from saved_frames directory
    saved_frames_dir = os.path.join(args.frame_dir, "saved_frames")
    os.makedirs(saved_frames_dir, exist_ok=True)
    
    frame_paths = os.listdir(args.frame_dir)

    for kf in frame_paths:
        if kf in keyframes:
            shutil.copy(os.path.join(args.frame_dir, kf), os.path.join(saved_frames_dir, kf))
            logger.info(f"Selected keyframe {kf} saved to {saved_frames_dir}")
