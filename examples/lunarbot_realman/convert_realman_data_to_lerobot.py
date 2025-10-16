"""
Efficient conversion script for RealMan manipulator data to LeRobot format.

This script converts data collected by realman_data_collection.py to LeRobot dataset format
for fine-tuning π₀.₅ models. It uses multiprocessing to efficiently handle large datasets
on multi-core systems.

Usage:
uv run examples/lunarbot_realman/convert_realman_data_to_lerobot.py --data_dir /path/to/realman_local_data

Optional arguments:
--push_to_hub: Push the converted dataset to Hugging Face Hub
--num_workers: Number of parallel workers (default: 80% of CPU cores)
--batch_size: Number of episodes to process per batch (default: 10)
"""

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm

REPO_NAME = "bingqiii/realman_manipulation"  # Change this to your desired dataset name
TARGET_IMAGE_SIZE = (256, 256)  # Target size for LeRobot

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for VLA training:
    1. Crop center square (using shorter edge as square size)
    2. Resize to 256x256 using high-quality interpolation
    3. Convert BGR to RGB
    
    Args:
        image: Input image in BGR format from cv2.imread
    
    Returns:
        Processed image in RGB format, shape (256, 256, 3)
    """
    # Convert BGR to RGB first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = image.shape[:2]
    
    # Determine square crop size (use shorter edge)
    crop_size = min(height, width)
    
    # Calculate crop coordinates for center crop
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size
    
    # Crop center square
    cropped = image[start_y:end_y, start_x:end_x]
    
    # Resize to 256x256 using high-quality interpolation
    # INTER_AREA is best for downsampling, maintains visual quality
    resized = cv2.resize(cropped, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    return resized

def load_episode_data(episode_path: Path) -> Tuple[Dict, List[Dict]]:
    """
    Load a single episode's data from disk.
    
    Returns:
        metadata: Episode metadata (task description)
        trajectory: List of step data with image paths converted to preprocessed arrays
    """
    # Load metadata
    with open(episode_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load trajectory
    with open(episode_path / "trajectory.json", "r") as f:
        trajectory = json.load(f)
    
    # Convert image paths to actual image arrays with preprocessing
    for step in trajectory:
        # Load and preprocess main camera image
        img_path = episode_path / step["image_path"]
        raw_image = cv2.imread(str(img_path))
        if raw_image is None:
            raise ValueError(f"Could not load image: {img_path}")
        step["image"] = preprocess_image(raw_image)
        
        # Load and preprocess wrist camera image
        wrist_img_path = episode_path / step["wrist_image_path"]
        raw_wrist_image = cv2.imread(str(wrist_img_path))
        if raw_wrist_image is None:
            raise ValueError(f"Could not load wrist image: {wrist_img_path}")
        step["wrist_image"] = preprocess_image(raw_wrist_image)
        
        # Convert state and action lists back to numpy arrays
        step["state"] = np.array(step["state"], dtype=np.float32)
        step["action"] = np.array(step["action"], dtype=np.float32)
    
    return metadata, trajectory

def process_episode_batch(args: Tuple[List[Path], str]) -> List[Tuple[Dict, List[Dict]]]:
    """
    Process a batch of episodes in parallel.
    This function will be executed in a separate process.
    """
    episode_paths, repo_name = args
    batch_data = []
    
    for episode_path in episode_paths:
        try:
            metadata, trajectory = load_episode_data(episode_path)
            batch_data.append((metadata, trajectory))
        except Exception as e:
            print(f"Error processing episode {episode_path}: {e}")
            continue
    
    return batch_data

def create_episode_batches(episode_paths: List[Path], batch_size: int) -> List[List[Path]]:
    """Create batches of episodes for parallel processing."""
    batches = []
    for i in range(0, len(episode_paths), batch_size):
        batch = episode_paths[i:i + batch_size]
        batches.append(batch)
    return batches

def main(
    data_dir: str, 
    *, 
    push_to_hub: bool = False,
    num_workers: int = None,
    batch_size: int = 10
):
    """
    Convert RealMan manipulator data to LeRobot format.
    
    Args:
        data_dir: Path to the directory containing episode folders
        push_to_hub: Whether to push the dataset to Hugging Face Hub
        num_workers: Number of parallel workers (default: 80% of CPU cores)
        batch_size: Number of episodes to process per batch
    """
    import multiprocessing as mp
    
    if num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.8))
    
    print(f"Using {num_workers} parallel workers with batch size {batch_size}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Get all episode directories
    episode_paths = sorted([p for p in data_path.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    print(f"Found {len(episode_paths)} episodes to convert")
    
    if len(episode_paths) == 0:
        print("No episodes found. Make sure your data directory contains episode_xxxx folders.")
        return
    
    # Inspect first episode to determine original image dimensions and verify preprocessing
    print("Inspecting first episode to determine data dimensions...")
    first_episode_path = episode_paths[0]
    
    # Load a sample image to check original dimensions
    with open(first_episode_path / "trajectory.json", "r") as f:
        sample_trajectory = json.load(f)
    
    if len(sample_trajectory) == 0:
        raise ValueError("First episode has no trajectory data")
    
    # Check original image dimensions
    sample_img_path = first_episode_path / sample_trajectory[0]["image_path"]
    original_image = cv2.imread(str(sample_img_path))
    if original_image is None:
        raise ValueError(f"Could not load sample image: {sample_img_path}")
    
    original_shape = original_image.shape
    print(f"Original image dimensions: {original_shape}")
    
    # Process the first episode to get processed dimensions
    sample_metadata, processed_trajectory = load_episode_data(episode_paths[0])
    sample_step = processed_trajectory[0]
    
    image_shape = sample_step["image"].shape
    wrist_image_shape = sample_step["wrist_image"].shape
    state_dim = len(sample_step["state"])
    action_dim = len(sample_step["action"])
    
    print(f"Processed data dimensions:")
    print(f"  Main image: {original_shape} → {image_shape}")
    print(f"  Wrist image: {original_shape} → {wrist_image_shape}")
    print(f"  State: {state_dim}")
    print(f"  Action: {action_dim}")
    
    # Verify we got the expected 256x256x3 shape
    expected_shape = (256, 256, 3)
    if image_shape != expected_shape or wrist_image_shape != expected_shape:
        raise ValueError(f"Image preprocessing failed. Expected {expected_shape}, got {image_shape} and {wrist_image_shape}")
    
    # Create LeRobot dataset with fixed 256x256x3 image features
    print("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="realman",
        fps=10,  # Based on COLLECTION_FPS in your data collection script
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image", 
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=min(20, num_workers),  # Limit image writer threads
        image_writer_processes=min(10, num_workers // 2),  # Limit image writer processes
    )
    
    # Create batches for parallel processing
    episode_batches = create_episode_batches(episode_paths, batch_size)
    print(f"Created {len(episode_batches)} batches for parallel processing")
    
    # Process episodes in parallel batches
    total_episodes_processed = 0
    total_steps_added = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_episode_batch, (batch, REPO_NAME)): i 
            for i, batch in enumerate(episode_batches)
        }
        
        # Process completed batches
        with tqdm(total=len(episode_batches), desc="Processing batches") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_data = future.result()
                    
                    # Add each episode from the batch to the dataset
                    for metadata, trajectory in batch_data:
                        if len(trajectory) == 0:
                            print(f"Skipping empty episode in batch {batch_idx}")
                            continue
                        
                        # Add all steps for this episode
                        for step in trajectory:
                            dataset.add_frame({
                                "image": step["image"],
                                "wrist_image": step["wrist_image"], 
                                "state": step["state"],
                                "actions": step["action"],
                                "task": metadata["task"],
                            })
                        
                        # Save the episode
                        dataset.save_episode()
                        total_episodes_processed += 1
                        total_steps_added += len(trajectory)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                
                pbar.update(1)
    
    print(f"\nConversion completed!")
    print(f"  Episodes processed: {total_episodes_processed}")
    print(f"  Total steps added: {total_steps_added}")
    print(f"  Average steps per episode: {total_steps_added / max(1, total_episodes_processed):.1f}")
    
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["realman", "manipulation", "robotics"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed to Hub successfully!")
    
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)