"""
Soccer Player Re-identification System
Implementation for Liat.ai Assignment

This module provides complete implementation for both assignment options:
1. Cross-camera player mapping
2. Single-feed re-identification

Author: Ronit Shrimankar
Email: ronitshrimankar1@gmail.com
Location: India
Date: 2025-06-13
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Player:
    """Player detection data structure"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    features: np.ndarray
    pose_keypoints: Optional[np.ndarray] = None
    jersey_number: Optional[int] = None
    team_color: Optional[str] = None
    timestamp: float = 0.0

@dataclass
class Track:
    """Player tracking data structure"""
    player_id: int
    history: deque
    last_seen: float
    feature_buffer: deque
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if not hasattr(self, 'history'):
            self.history = deque(maxlen=30)
        if not hasattr(self, 'feature_buffer'):
            self.feature_buffer = deque(maxlen=10)

class FeatureExtractor:
    """Enhanced feature extraction for player re-identification"""
    
    def __init__(self, feature_dim=512):
        self.feature_dim = feature_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_backbone()
        
    def _init_backbone(self):
        """Initialize feature extraction backbone"""
        # Using ResNet50 as feature extractor
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.feature_dim)
        self.backbone.to(self.device)
        self.backbone.eval()
        
        # Preprocessing
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_appearance_features(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract appearance features from player crop"""
        if image_crop.size == 0:
            return np.zeros(self.feature_dim)
            
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
            # Preprocess and extract features
            input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.backbone(input_tensor)
                features = F.normalize(features, p=2, dim=1)
                
            return features.cpu().numpy().flatten()
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return np.zeros(self.feature_dim)
    
    def extract_pose_features(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract pose-based features (simplified implementation)"""
        # Simplified pose feature extraction
        # In production, would use MediaPipe or OpenPose
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
        
        # Extract simple geometric features
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = gray.shape[1] // 2, gray.shape[0] // 2
            
        # Basic shape descriptors
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            aspect_ratio = gray.shape[1] / gray.shape[0] if gray.shape[0] > 0 else 1.0
        else:
            area, perimeter, aspect_ratio = 0, 0, 1.0
            
        return np.array([cx, cy, area, perimeter, aspect_ratio])
    
    def extract_color_features(self, image_crop: np.ndarray) -> np.ndarray:
        """Extract dominant color features for team classification"""
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)
        
        return np.concatenate([hist_h, hist_s, hist_v])

class PlayerReIDSystem:
    """Main player re-identification system"""
    
    def __init__(self, model_path: str = "data/best.pt"):
        self.model_path = model_path
        self.feature_extractor = FeatureExtractor()
        self.detection_model = None
        self.tracks = {}
        self.next_id = 1
        self.similarity_threshold = 0.7
        self.max_lost_frames = 30
        
        self._load_detection_model()
        
    def _load_detection_model(self):
        """Load YOLOv11 detection model"""
        try:
            if os.path.exists(self.model_path):
                self.detection_model = YOLO(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                # Fallback to YOLOv11 pretrained model
                self.detection_model = YOLO('yolo11n.pt')
                logger.warning(f"Model not found at {self.model_path}, using pretrained YOLOv11")
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            self.detection_model = YOLO('yolo11n.pt')
    
    def detect_players(self, frame: np.ndarray, timestamp: float = 0.0) -> List[Player]:
        """Detect players in frame using YOLOv11"""
        if self.detection_model is None:
            return []
            
        results = self.detection_model(frame, verbose=False)
        players = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Filter for person class (assuming class 0 is person)
                if int(box.cls) == 0 and float(box.conf) > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf)
                    
                    # Extract player crop
                    player_crop = frame[y1:y2, x1:x2]
                    if player_crop.size == 0:
                        continue
                    
                    # Extract features
                    appearance_features = self.feature_extractor.extract_appearance_features(player_crop)
                    pose_features = self.feature_extractor.extract_pose_features(player_crop)
                    color_features = self.feature_extractor.extract_color_features(player_crop)
                    
                    # Combine features
                    combined_features = np.concatenate([
                        appearance_features, 
                        pose_features, 
                        color_features
                    ])
                    
                    player = Player(
                        id=-1,  # Will be assigned during tracking
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        features=combined_features,
                        timestamp=timestamp
                    )
                    players.append(player)
        
        return players
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between feature vectors"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Split features into components
        app_dim = 512  # Appearance features dimension
        pose_dim = 5   # Pose features dimension
        
        app_feat1, pose_feat1 = features1[:app_dim], features1[app_dim:app_dim+pose_dim]
        app_feat2, pose_feat2 = features2[:app_dim], features2[app_dim:app_dim+pose_dim]
        
        # Calculate weighted similarity
        app_sim = 1 - cosine(app_feat1, app_feat2) if len(app_feat1) > 0 else 0
        pose_sim = 1 - cosine(pose_feat1, pose_feat2) if len(pose_feat1) > 0 else 0
        
        # Weighted combination (appearance gets higher weight)
        combined_sim = 0.7 * app_sim + 0.3 * pose_sim
        return max(0, combined_sim)
    
    def update_tracks(self, players: List[Player], timestamp: float) -> List[Player]:
        """Update player tracks with new detections"""
        if not players:
            return []
        
        # Calculate similarity matrix
        active_tracks = {tid: track for tid, track in self.tracks.items() 
                        if timestamp - track.last_seen <= self.max_lost_frames}
        
        if not active_tracks:
            # Initialize new tracks
            for player in players:
                player.id = self.next_id
                track = Track(
                    player_id=self.next_id,
                    history=deque([player], maxlen=30),
                    feature_buffer=deque([player.features], maxlen=10),
                    last_seen=timestamp,
                    confidence_score=player.confidence
                )
                self.tracks[self.next_id] = track
                self.next_id += 1
            return players
        
        # Create similarity matrix
        track_ids = list(active_tracks.keys())
        similarity_matrix = np.zeros((len(players), len(track_ids)))
        
        for i, player in enumerate(players):
            for j, track_id in enumerate(track_ids):
                track = active_tracks[track_id]
                # Use average features from buffer
                avg_features = np.mean(list(track.feature_buffer), axis=0)
                similarity_matrix[i, j] = self.calculate_similarity(player.features, avg_features)
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        assigned_players = []
        assigned_tracks = set()
        
        for i, j in zip(row_ind, col_ind):
            if similarity_matrix[i, j] > self.similarity_threshold:
                track_id = track_ids[j]
                players[i].id = track_id
                
                # Update track
                track = self.tracks[track_id]
                track.history.append(players[i])
                track.feature_buffer.append(players[i].features)
                track.last_seen = timestamp
                track.confidence_score = 0.8 * track.confidence_score + 0.2 * players[i].confidence
                
                assigned_players.append(players[i])
                assigned_tracks.add(track_id)
        
        # Handle unassigned players (new tracks)
        for i, player in enumerate(players):
            if i not in row_ind or similarity_matrix[i, col_ind[list(row_ind).index(i)]] <= self.similarity_threshold:
                player.id = self.next_id
                track = Track(
                    player_id=self.next_id,
                    history=deque([player], maxlen=30),
                    feature_buffer=deque([player.features], maxlen=10),
                    last_seen=timestamp,
                    confidence_score=player.confidence
                )
                self.tracks[self.next_id] = track
                assigned_players.append(player)
                self.next_id += 1
        
        return assigned_players

class CrossCameraMapper:
    """Cross-camera player mapping system"""
    
    def __init__(self):
        self.reid_system1 = PlayerReIDSystem()
        self.reid_system2 = PlayerReIDSystem()
        self.cross_camera_mapping = {}
        
    def process_videos(self, video1_path: str, video2_path: str) -> Dict:
        """Process both videos and create cross-camera mapping"""
        results = {
            'video1_tracks': {},
            'video2_tracks': {},
            'cross_camera_mapping': {},
            'sync_info': {}
        }
        
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        frame_count = 0
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                break
            
            timestamp = frame_count / 30.0  # Assuming 30 FPS
            
            # Process both frames
            players1 = self.reid_system1.detect_players(frame1, timestamp)
            players2 = self.reid_system2.detect_players(frame2, timestamp)
            
            tracked_players1 = self.reid_system1.update_tracks(players1, timestamp)
            tracked_players2 = self.reid_system2.update_tracks(players2, timestamp)
            
            # Store results
            results['video1_tracks'][frame_count] = [
                {
                    'player_id': p.id,
                    'bbox': p.bbox,
                    'confidence': p.confidence
                } for p in tracked_players1
            ]
            
            results['video2_tracks'][frame_count] = [
                {
                    'player_id': p.id,
                    'bbox': p.bbox,
                    'confidence': p.confidence
                } for p in tracked_players2
            ]
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap1.release()
        cap2.release()
        
        # Create cross-camera mapping
        results['cross_camera_mapping'] = self._create_cross_mapping()
        
        return results
    
    def _create_cross_mapping(self) -> Dict:
        """Create mapping between player IDs across cameras"""
        # Extract features from both tracking systems
        features1 = {}
        features2 = {}
        
        for track_id, track in self.reid_system1.tracks.items():
            if len(track.feature_buffer) > 0:
                features1[track_id] = np.mean(list(track.feature_buffer), axis=0)
        
        for track_id, track in self.reid_system2.tracks.items():
            if len(track.feature_buffer) > 0:
                features2[track_id] = np.mean(list(track.feature_buffer), axis=0)
        
        # Calculate cross-camera similarity matrix
        mapping = {}
        used_ids2 = set()
        
        for id1, feat1 in features1.items():
            best_match = None
            best_similarity = 0.6  # Threshold for cross-camera matching
            
            for id2, feat2 in features2.items():
                if id2 in used_ids2:
                    continue
                    
                similarity = 1 - cosine(feat1, feat2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = id2
            
            if best_match is not None:
                mapping[id1] = best_match
                used_ids2.add(best_match)
        
        return mapping

class SingleFeedReID:
    """Single feed re-identification system"""
    
    def __init__(self):
        self.reid_system = PlayerReIDSystem()
        
    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """Process single video with re-identification"""
        results = {
            'tracks': {},
            'player_statistics': {},
            'tracking_quality': {}
        }
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # Setup video writer if output path provided
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / 30.0
            
            # Detect and track players
            players = self.reid_system.detect_players(frame, timestamp)
            tracked_players = self.reid_system.update_tracks(players, timestamp)
            
            # Store results
            results['tracks'][frame_count] = [
                {
                    'player_id': p.id,
                    'bbox': p.bbox,
                    'confidence': p.confidence,
                    'timestamp': timestamp
                } for p in tracked_players
            ]
            
            # Draw tracking results on frame
            if out:
                annotated_frame = self._draw_tracks(frame, tracked_players)
                out.write(annotated_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        if out:
            out.release()
        
        # Calculate statistics
        results['player_statistics'] = self._calculate_statistics()
        results['tracking_quality'] = self._assess_tracking_quality()
        
        return results
    
    def _draw_tracks(self, frame: np.ndarray, players: List[Player]) -> np.ndarray:
        """Draw tracking results on frame"""
        annotated = frame.copy()
        
        for player in players:
            x1, y1, x2, y2 = player.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw player ID
            cv2.putText(annotated, f"ID: {player.id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(annotated, f"{player.confidence:.2f}", (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return annotated
    
    def _calculate_statistics(self) -> Dict:
        """Calculate tracking statistics"""
        stats = {}
        
        for track_id, track in self.reid_system.tracks.items():
            stats[track_id] = {
                'total_detections': len(track.history),
                'avg_confidence': track.confidence_score,
                'track_duration': len(track.history) / 30.0,  # Assuming 30 FPS
                'last_seen': track.last_seen
            }
        
        return stats
    
    def _assess_tracking_quality(self) -> Dict:
        """Assess overall tracking quality"""
        total_tracks = len(self.reid_system.tracks)
        avg_confidence = np.mean([t.confidence_score for t in self.reid_system.tracks.values()])
        
        return {
            'total_unique_players': total_tracks,
            'average_confidence': avg_confidence,
            'tracking_stability': self._calculate_stability()
        }
    
    def _calculate_stability(self) -> float:
        """Calculate tracking stability metric"""
        if not self.reid_system.tracks:
            return 0.0
        
        stabilities = []
        for track in self.reid_system.tracks.values():
            if len(track.history) > 1:
                # Calculate bbox consistency
                bboxes = [p.bbox for p in track.history]
                areas = [(bbox[2]-bbox[0])*(bbox[3]-bbox[1]) for bbox in bboxes]
                area_stability = 1 - (np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else 0
                stabilities.append(max(0, area_stability))
        
        return np.mean(stabilities) if stabilities else 0.0

def main():
    """Main function to demonstrate the system"""
    logger.info("Soccer Player Re-identification System")
    logger.info("=====================================")
    
    # Option 1: Cross-camera mapping example
    logger.info("Option 1: Cross-camera player mapping")
    cross_mapper = CrossCameraMapper()
    
    # Option 2: Single feed re-identification example  
    logger.info("Option 2: Single feed re-identification")
    single_reid = SingleFeedReID()
    
    logger.info("System initialized successfully!")
    
    # Example usage (would need actual video files)
    """
    # For cross-camera mapping:
    results = cross_mapper.process_videos('broadcast.mp4', 'tacticam.mp4')
    
    # For single feed:
    results = single_reid.process_video('15sec_input_729p.mp4', 'output_tracked.mp4')
    """

if __name__ == "__main__":
    main()
