#!/usr/bin/env python3
"""
Option 1: Cross-Camera Player Mapping
Implementation for Liat.ai Assignment

This script implements cross-camera player mapping between broadcast.mp4 and tacticam.mp4
using the provided YOLOv11 model and advanced re-identification techniques.

Usage:
    python option1_cross_camera.py --broadcast broadcast.mp4 --tacticam tacticam.mp4 --output results.json

Author: Ronit Shrimankar
Email: ronitshrimankar1@gmail.com
Location: India
Date: 2025-06-13
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from player_reid_system import CrossCameraMapper
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cross_camera_mapping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrossCameraPipeline:
    """Complete pipeline for cross-camera player mapping"""
    
    def __init__(self, model_path: str = "data/best.pt"):
        self.model_path = model_path
        self.mapper = CrossCameraMapper()
        
    def validate_inputs(self, broadcast_path: str, tacticam_path: str) -> bool:
        """Validate input video files"""
        if not os.path.exists(broadcast_path):
            logger.error(f"Broadcast video not found: {broadcast_path}")
            return False
            
        if not os.path.exists(tacticam_path):
            logger.error(f"Tacticam video not found: {tacticam_path}")
            return False
            
        logger.info("Input videos validated successfully")
        return True
    
    def process_videos(self, broadcast_path: str, tacticam_path: str, 
                      output_path: str) -> bool:
        """Main processing pipeline"""
        try:
            logger.info("Starting cross-camera player mapping...")
            logger.info(f"Broadcast video: {broadcast_path}")
            logger.info(f"Tacticam video: {tacticam_path}")
            
            start_time = time.time()
            
            # Process videos and create mapping
            results = self.mapper.process_videos(broadcast_path, tacticam_path)
            
            # Add metadata
            results['metadata'] = {
                'broadcast_video': broadcast_path,
                'tacticam_video': tacticam_path,
                'processing_time': time.time() - start_time,
                'model_used': self.model_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Processing completed in {results['metadata']['processing_time']:.2f} seconds")
            
            # Print summary
            self._print_summary(results)
            
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return False
    
    def _print_summary(self, results: dict):
        """Print processing summary"""
        logger.info("\n" + "="*50)
        logger.info("CROSS-CAMERA MAPPING SUMMARY")
        logger.info("="*50)
        
        # Video 1 stats
        total_frames_v1 = len(results['video1_tracks'])
        unique_players_v1 = len(set([p['player_id'] for frame_players in results['video1_tracks'].values() 
                                   for p in frame_players]))
        
        # Video 2 stats  
        total_frames_v2 = len(results['video2_tracks'])
        unique_players_v2 = len(set([p['player_id'] for frame_players in results['video2_tracks'].values() 
                                   for p in frame_players]))
        
        # Cross-camera mapping stats
        mapped_players = len(results['cross_camera_mapping'])
        
        logger.info(f"Broadcast Video (Video 1):")
        logger.info(f"  - Total frames processed: {total_frames_v1}")
        logger.info(f"  - Unique players detected: {unique_players_v1}")
        
        logger.info(f"Tacticam Video (Video 2):")
        logger.info(f"  - Total frames processed: {total_frames_v2}")
        logger.info(f"  - Unique players detected: {unique_players_v2}")
        
        logger.info(f"Cross-Camera Mapping:")
        logger.info(f"  - Successfully mapped players: {mapped_players}")
        logger.info(f"  - Mapping ratio: {mapped_players/max(unique_players_v1, unique_players_v2)*100:.1f}%")
        
        logger.info("\nPlayer ID Mappings:")
        for v1_id, v2_id in results['cross_camera_mapping'].items():
            logger.info(f"  Broadcast Player {v1_id} ↔ Tacticam Player {v2_id}")
        
        logger.info("="*50)

def create_sample_output():
    """Create sample output structure for demonstration"""
    sample_output = {
        "video1_tracks": {
            "0": [
                {"player_id": 1, "bbox": [100, 200, 150, 300], "confidence": 0.85},
                {"player_id": 2, "bbox": [300, 180, 350, 280], "confidence": 0.92}
            ],
            "30": [
                {"player_id": 1, "bbox": [105, 205, 155, 305], "confidence": 0.88},
                {"player_id": 2, "bbox": [295, 175, 345, 275], "confidence": 0.89}
            ]
        },
        "video2_tracks": {
            "0": [
                {"player_id": 1, "bbox": [200, 150, 250, 250], "confidence": 0.87},
                {"player_id": 2, "bbox": [400, 120, 450, 220], "confidence": 0.91}
            ],
            "30": [
                {"player_id": 1, "bbox": [205, 155, 255, 255], "confidence": 0.84},
                {"player_id": 2, "bbox": [395, 125, 445, 225], "confidence": 0.93}
            ]
        },
        "cross_camera_mapping": {
            "1": 1,  # Broadcast Player 1 maps to Tacticam Player 1
            "2": 2   # Broadcast Player 2 maps to Tacticam Player 2
        },
        "metadata": {
            "broadcast_video": "broadcast.mp4",
            "tacticam_video": "tacticam.mp4",
            "processing_time": 45.2,
            "model_used": "data/best.pt",
            "timestamp": "2025-06-13 00:39:10"
        }
    }
    
    return sample_output

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Cross-Camera Player Mapping')
    parser.add_argument('--broadcast', required=True, help='Path to broadcast video')
    parser.add_argument('--tacticam', required=True, help='Path to tacticam video')
    parser.add_argument('--output', default='cross_camera_results.json', 
                       help='Output JSON file path')
    parser.add_argument('--model', default='data/best.pt', 
                       help='Path to YOLOv11 model')
    parser.add_argument('--demo', action='store_true', 
                       help='Generate demo output without processing videos')
    
    args = parser.parse_args()
    
    if args.demo:
        logger.info("Generating demonstration output...")
        sample_results = create_sample_output()
        with open(args.output, 'w') as f:
            json.dump(sample_results, f, indent=2)
        logger.info(f"Demo results saved to: {args.output}")
        return
    
    # Initialize pipeline
    pipeline = CrossCameraPipeline(args.model)
    
    # Validate inputs
    if not pipeline.validate_inputs(args.broadcast, args.tacticam):
        sys.exit(1)
    
    # Process videos
    success = pipeline.process_videos(args.broadcast, args.tacticam, args.output)
    
    if success:
        logger.info("Cross-camera mapping completed successfully!")
        sys.exit(0)
    else:
        logger.error("Cross-camera mapping failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
