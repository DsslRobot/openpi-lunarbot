"""
Script to collect data from a RealMan manipulator and two RealSense cameras,
and save it to the local disk.

This script uses a multi-threaded, timestamp-based approach to synchronize and
capture data from the robot and cameras concurrently.

Usage:
1. Make sure the robot and cameras are connected.
2. Run the script: `uv run examples/lunarbot_realman/realman_data_to_disk.py`
3. Follow the prompts in the terminal to start and stop data collection for each episode.
"""

import shutil
import time
import threading
import json
from pathlib import Path
import collections

import numpy as np
import cv2
import pyrealsense2 as rs
import tyro
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

# MACROS
## Manipulator
ARM_IP = "192.168.1.17"
ARM_PORT = 8080

## Realsense Cameras Params
# Find serial numbers using `realsense-viewer` or `rs-enumerate-devices`
WRIST_CAM_SERIAL = "215422254539"  # Replace with your wrist camera's serial number
THIRD_VIEW_CAM_SERIAL = "145422251468" # Replace with your 3rd-person camera's serial number
IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_FPS = 30
TARGET_IMG_SIZE = (256, 256)

## Data Saving Params
DATA_ROOT_DIR = "realman_local_data"
COLLECTION_FPS = 10
# Max allowed time difference (in seconds) between camera and robot data for a valid sync.
# A good starting point is half the collection interval.
TIMESTAMP_TOLERANCE = (1.0 / COLLECTION_FPS) / 2.0

# SHARED DATA STRUCTURES
# Use deques as thread-safe buffers with a max length
camera_buffer = collections.deque(maxlen=10)
robot_state_buffer = collections.deque(maxlen=10)
action_buffer = collections.deque(maxlen=10)
stop_event = threading.Event()


def camera_worker(wrist_serial: str, third_view_serial: str):
    """
    Initializes and continuously fetches frames from two RealSense cameras.
    Adds a timestamp, resizes frames, and appends to a shared buffer.
    """
    # Initialize pipelines
    pipelines = []
    serials = [wrist_serial, third_view_serial]
    for serial in serials:
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, IMG_FPS)
        pipe.start(config)
        pipelines.append(pipe)
    
    print("Camera thread started.")
    while not stop_event.is_set():
        try:
            frames = [p.wait_for_frames(timeout_ms=1000) for p in pipelines]
            timestamp = time.time() # Get timestamp as close to frame arrival as possible
            color_frames = [f.get_color_frame() for f in frames]

            if not all(color_frames):
                continue

            images = [np.asanyarray(cf.get_data()) for cf in color_frames]
            wrist_resized = cv2.resize(images[0], TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
            third_view_resized = cv2.resize(images[1], TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)

            camera_buffer.append({
                "timestamp": timestamp,
                "wrist_image": wrist_resized,
                "image": third_view_resized,
            })

        except RuntimeError as e:
            print(f"Camera thread error: {e}")
            time.sleep(1)
            continue
    
    for p in pipelines:
        p.stop()
    print("Camera thread stopped.")


def robot_state_worker(arm: RoboticArm):
    """
    Continuously fetches the robot's joint angles and gripper state.
    Adds a timestamp and appends to a shared buffer.
    """
    print("Robot state thread started.")
    while not stop_event.is_set():
        try:
            ret_joint, joint_angles_deg = arm.rm_get_joint_degree()
            timestamp = time.time()
            if ret_joint != 0: continue
            
            ret_gripper, gripper_state = arm.rm_get_gripper_state()
            if ret_gripper != 0: continue

            gripper_open = 1.0 if gripper_state.get("mode") == 1 else 0.0
            joint_angles_rad = np.deg2rad(np.array(joint_angles_deg))
            arm_state = np.concatenate((joint_angles_rad, [gripper_open]))

            robot_state_buffer.append({"timestamp": timestamp, "state": arm_state})

        except Exception as e:
            print(f"Robot state thread error: {e}")
        
        time.sleep(0.01)
    print("Robot state thread stopped.")


def action_worker(arm: RoboticArm):
    """
    Continuously fetches the robot's target end-effector pose and gripper state.
    Adds a timestamp and appends to a shared buffer.
    """
    print("Action worker thread started.")
    while not stop_event.is_set():
        try:
            ret, arm_state_dict = arm.rm_get_current_arm_state()
            timestamp = time.time()
            if ret != 0 or not arm_state_dict: continue
            
            pose = arm_state_dict.get('pose')
            if not pose: continue

            xyz = pose.position
            rpy = pose.euler
            pose_6d = np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]])

            ret_gripper, gripper_state = arm.rm_get_gripper_state()
            if ret_gripper != 0: continue
            gripper_open = 1.0 if gripper_state.get('mode') == 1 else 0.0

            action_7d = np.concatenate((pose_6d, [gripper_open]))

            action_buffer.append({"timestamp": timestamp, "action": action_7d})

        except Exception as e:
            print(f"Action worker thread error: {e}")

        time.sleep(0.01)
    print("Action worker thread stopped.")


def find_closest_in_buffer(buffer, target_timestamp):
    """Finds the item in a buffer with the timestamp closest to the target."""
    if not buffer:
        return None, float('inf')
    
    # Since deques are not indexable, we convert to a list for searching
    buffer_list = list(buffer)
    
    closest_item = min(buffer_list, key=lambda x: abs(x['timestamp'] - target_timestamp))
    min_diff = abs(closest_item['timestamp'] - target_timestamp)
    
    return closest_item, min_diff


def main():
    """
    Main function to orchestrate data collection.
    """
    # --- 1. Initialize Output Directory ---
    output_path = Path(DATA_ROOT_DIR)
    if output_path.exists():
        print(f"Dataset directory already exists at {output_path}. Deleting it.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    print(f"Data will be saved in: {output_path.resolve()}")
    episode_idx = 0

    # --- 2. Initialize Robot Arm ---
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ARM_IP, ARM_PORT)
    if handle.id == -1:
        print(f"Failed to connect to robot arm at {ARM_IP}:{ARM_PORT}")
        return
    print(f"Successfully connected to robot arm (handle: {handle.id}).")

    # --- 3. Start Data Collection Loop ---
    try:
        while True:
            language_instruction = input("\n>>> Enter the task description for the next episode (e.g., 'pick up the red block'), or 'q' to quit: ")
            if language_instruction.lower() == 'q': break

            # Create directories for the new episode
            episode_path = output_path / f"episode_{episode_idx:04d}"
            img_path = episode_path / "images"
            wrist_img_path = episode_path / "wrist_images"
            img_path.mkdir(parents=True)
            wrist_img_path.mkdir()

            # Save metadata
            with open(episode_path / "metadata.json", "w") as f:
                json.dump({"task": language_instruction}, f, indent=4)

            input(">>> Press [Enter] to START recording this episode...")

            stop_event.clear()
            camera_buffer.clear()
            robot_state_buffer.clear()
            action_buffer.clear()

            cam_thread = threading.Thread(target=camera_worker, args=(WRIST_CAM_SERIAL, THIRD_VIEW_CAM_SERIAL))
            robot_thread = threading.Thread(target=robot_state_worker, args=(arm,))
            action_thread = threading.Thread(target=action_worker, args=(arm,))
            cam_thread.start()
            robot_thread.start()
            action_thread.start()

            time.sleep(2)
            print(">>> Recording... Press [Enter] to STOP.")
            
            episode_data = []
            last_save_time = time.time()
            sync_misses = 0

            # Main collection loop with timestamp synchronization
            while not stop_event.is_set():
                if time.time() - last_save_time < 1.0 / COLLECTION_FPS:
                    time.sleep(0.001)
                    continue
                
                last_save_time = time.time()

                try:
                    # Use camera as the pacemaker
                    cam_data = camera_buffer.popleft()
                except IndexError:
                    continue # Camera buffer is empty, wait for a frame

                cam_ts = cam_data['timestamp']

                # Find the closest state and action in their respective buffers
                state_data, state_diff = find_closest_in_buffer(robot_state_buffer, cam_ts)
                action_data, action_diff = find_closest_in_buffer(action_buffer, cam_ts)

                # Check if both are within the tolerance
                if state_data and action_data and state_diff < TIMESTAMP_TOLERANCE and action_diff < TIMESTAMP_TOLERANCE:
                    step_data = {
                        "image": cam_data["image"],
                        "wrist_image": cam_data["wrist_image"],
                        "state": state_data["state"],
                        "action": action_data["action"],
                        "timestamp": cam_ts,
                    }
                    episode_data.append(step_data)
                    print(f"\rCollected step {len(episode_data)} (sync misses: {sync_misses})", end="")
                else:
                    sync_misses += 1

                if msvcrt.kbhit() and msvcrt.getch() == b'\r':
                    stop_event.set()

            # --- 4. Save the Episode ---
            print(f"\nRecording stopped. Collected {len(episode_data)} steps for episode.")
            
            cam_thread.join()
            robot_thread.join()
            action_thread.join()

            if len(episode_data) < 1:
                print("Episode has no data, not saving.")
                shutil.rmtree(episode_path)
                continue

            print(f"Saving episode to {episode_path}...")
            
            # Prepare data for bulk saving
            states = []
            actions = []
            timestamps = []
            
            for i, step in enumerate(episode_data):
                # Save images
                cv2.imwrite(str(img_path / f"{i:04d}.png"), step["image"])
                cv2.imwrite(str(wrist_img_path / f"{i:04d}.png"), step["wrist_image"])
                
                # Collect numpy data
                states.append(step["state"])
                actions.append(step["action"])
                timestamps.append(step["timestamp"])

            # Save all numpy data in a single compressed file
            np.savez_compressed(
                episode_path / "trajectory.npz",
                state=np.array(states),
                action=np.array(actions),
                timestamp=np.array(timestamps),
            )
            
            print("Episode saved successfully.")
            episode_idx += 1

    finally:
        # --- 5. Release Resources ---
        print("Cleaning up resources...")
        stop_event.set()
        RoboticArm.rm_destroy()
        print("Script finished.")


if __name__ == "__main__":
    try:
        import msvcrt
    except ImportError:
        import sys, tty, termios, select
        print("Warning: Non-Windows OS detected. Stopping requires pressing Enter and might be less responsive.")
        class Msvcrt:
            def kbhit(self):
                dr,dw,de = select.select([sys.stdin], [], [], 0)
                return dr != []
            def getch(self):
                return sys.stdin.read(1)
        msvcrt = Msvcrt()

    tyro.cli(main)