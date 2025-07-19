# Quest3-Teleop-in-Isaac-Gym

# 📚Overview

This project enables immersive teleoperation of dual **Franka Panda robotic arms** equipped with **Inspire Hands** using **Meta Quest 3** in a **NVIDIA Isaac Gym** physics simulation environment. The system provides real-time bilateral control with advanced hand gesture recognition and multi-modal visual feedback.

<p align="center">
  <img
    src="repo_assets/demo.webp"
    controls
    muted
    style="max-width: 100%; height: auto;">
  </video>
</p>


### Key Features

- **VR Teleoperation**: Use Meta Quest 3 for natural hand-based control of dual robotic arms
- **Real-time Streaming**: Low-latency video streaming from simulation to VR headset
- **Dual Arm Control**: Simultaneous control of left and right Franka Panda arms with Inspire Hands
- **Advanced IK Solving**: Real-time inverse kinematics using Pink/Pinocchio libraries
- **Hand Retargeting**: Accurate human-to-robot hand pose mapping using dex-retargeting
- **Gesture Control**: Intuitive gesture-based commands for recording and visualization modes
- **Multi-Modal Visualization**: RGB, depth, and segmentation mask display modes
- **Episode Recording**: Gesture-controlled recording system for data collection

### Technical Architecture

- **Frontend**: WebRTC-based VR streaming interface for Quest 3
- **Physics Engine**: Isaac Gym with GPU-accelerated PhysX simulation
- **Kinematics**: Pink (Pinocchio-based) inverse kinematics solver
- **Hand Tracking**: Vuer-based hand pose processing and retargeting
- **Visual Processing**: Real-time depth colorization and segmentation visualization

# 🛠️Installation

### System Requirements

The operating system requirement is **Ubuntu 20.04** or higher. This project has been developed and tested on **Ubuntu 20.04**, but should work on newer Ubuntu versions (22.04) as well.

**Prerequisites:**
- NVIDIA GPU with CUDA support (required for Isaac Gym)
- CUDA toolkit (compatible with your GPU driver)

### Setup Instructions

1. **Install OpenTeleVision Environment**

   Create a new conda environment with Python 3.8:
   ```bash
   conda create -n teleop python=3.8
   conda activate teleop
   ```

   Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Isaac Gym**
   
   Download and install Isaac Gym from NVIDIA:
   - Visit: https://developer.nvidia.com/isaac-gym/
   - Register and download the Isaac Gym Preview package
   - Follow the installation instructions in the Isaac Gym documentation
   
   **Note:** Isaac Gym requires a compatible NVIDIA GPU and proper CUDA installation.

### Important Notes

- The **numpy** version required by **dex-retargeting-0.4.6** may have compatibility considerations with **Pinocchio**. If you encounter version conflicts, the project has been tested with **numpy>=1.24.4**.
- Ensure your NVIDIA drivers are up to date and compatible with your CUDA installation before installing Isaac Gym.

## :computer:Usage

### Quest 3 Setup and Connection

#### Prerequisites
- Meta Quest 3 headset
- USB-C cable (for connecting Quest 3 to computer)
- Android SDK Platform Tools

#### Setup Steps

1. **Install Android SDK Platform Tools**
   
   Download and install the Android SDK Platform Tools:
   - Visit: https://developer.android.com/tools/releases/platform-tools
   - Download the appropriate package for Linux
   - Extract the package to your desired location (e.g., `~/Android_SDK_Platform_Tools/`)

2. **Configure ADB Environment**
   
   Add ADB tools to your system PATH by modifying `~/.bashrc`:
   ```bash
   echo 'export PATH=$PATH:~/Android_SDK_Platform_Tools/platform-tools-latest-linux/platform-tools/' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   **Note:** Adjust the path according to your actual installation directory.

3. **Connect Quest 3 to Computer**
   
   - Connect your Quest 3 to the computer using a USB-C cable
   - Put on the Quest 3 headset
   - Accept the USB debugging permission prompt that appears in VR
   
4. **Verify Connection**
   
   Test the connection by running:
   ```bash
   adb devices
   ```
   
   You should see output similar to:
   ```
   List of devices attached
   1WMHH123456789  device
   ```

5. **Set Up Port Forwarding**
   
   Configure port forwarding for the teleoperation interface:
   ```bash
   adb reverse tcp:8012 tcp:8012
   ```
   
   This allows the Quest 3 to access the teleoperation server running on your computer. 

### Teleoperation in Isaac Gym

#### Quick Start

1. **Launch the Teleoperation System**
   ```bash
   python teleop_run.py
   ```
   This will start the Isaac Gym simulation with default parameters. You should see:
   - Isaac Gym simulation window displaying dual Franka Panda arms with Inspire Hands
   - A table with manipulatable objects in the scene
   - Console output showing successful URDF loading and physics initialization

2. **Connect Quest 3 to VR Interface**
   - Put on your Quest 3 headset
   - Open the browser and navigate to `http://127.0.0.1:8012/`
   - Click **"Enter VR"** button to start the immersive teleoperation session

3. **Start Teleoperation**
   - Your hand movements will be tracked and mapped to the robotic arms in real-time
   - Left hand controls the left Franka arm, right hand controls the right Franka arm
   - Hand gestures are automatically retargeted to the Inspire Hand finger movements

#### Advanced Usage with Custom Parameters

For more control over the system, you have two options:

**Option 1: Use the provided bash script (Recommended)**
```bash
./run_teleop.sh
```
The project includes a pre-configured `run_teleop.sh` script with optimized parameters. You can edit this script to customize the settings:

**Option 2: Use command line options directly**

```bash
python teleop_run.py \
    --isaac_output "output/my_session" \
    --robot_ik_urdf "assets/franka_pinocchio/robots/franka_panda.urdf" \
    --assets_path "assets/" \
    --init_mode "rgb" \
    --vis_camera true \
    --vis_issac true \
    --ik_dt 0.01 \
    --ik_thresh 0.01 \
    --ik_max_iterate_step 100
```

**Parameter Descriptions:**
- `--isaac_output`: Directory to save recorded episodes and videos
- `--robot_ik_urdf`: Path to the robot URDF file for inverse kinematics
- `--assets_path`: Root directory containing robot and hand assets
- `--init_mode`: Initial visualization mode (`rgb`, `depth`, `mask`)
- `--vis_camera`: Enable real-time pygame visualization window
- `--vis_issac`: Enable Isaac Gym viewer window
- `--ik_dt`: Time step for inverse kinematics solver (smaller = more accurate)
- `--ik_thresh`: Convergence threshold for IK solver
- `--ik_max_iterate_step`: Maximum iterations for IK solver

#### Gesture Controls

**Right Hand Gestures (Display Mode Control):**
- 👍🏻 **Thumb + Index**: Switch to depth visualization mode
- 👍🏻 **Thumb + Middle**: Switch to segmentation mask mode  
- 👍🏻 **Thumb + Ring**: Switch to RGB color mode

**Left Hand Gestures (Recording Control):**
- 👍🏻 **Thumb + Index**: Start recording new episode
- 👍🏻 **Thumb + Middle**: Stop current episode recording

#### Real-time Visualization

The system provides three visualization modes:

1. **RGB Mode**: Natural color view of the simulation
2. **Depth Mode**: Colored depth maps showing distance information
3. **Segmentation Mode**: Color-coded object and robot part identification

#### Episode Recording

- Episodes are automatically saved in timestamped directories under the output folder
- Each episode contains RGB, depth, and segmentation videos for both cameras
- Videos are saved at 30 FPS in MP4 format

**Directory Structure:**
```
output/
└── 20250719_132924/              # Session timestamp
    ├── episode1/
    │   ├── RGB/
    │   │   ├── left_rgb.mp4      # Left camera RGB video
    │   │   └── right_rgb.mp4     # Right camera RGB video
    │   ├── Depth/
    │   │   ├── left_depth.mp4    # Left camera colored depth video
    │   │   └── right_depth.mp4   # Right camera colored depth video
    │   └── Segment/
    │       ├── left_segment.mp4  # Left camera segmentation video
    │       └── right_segment.mp4 # Right camera segmentation video
    ├── episode2/
    │   ├── RGB/
    │   │   ├── left_rgb.mp4
    │   │   └── right_rgb.mp4
    │   ├── Depth/
    │   │   ├── left_depth.mp4
    │   │   └── right_depth.mp4
    │   └── Segment/
    │       ├── left_segment.mp4
    │       └── right_segment.mp4
    └── episode3/
        └── ...
```

#### Troubleshooting

**Common Issues:**
- If Isaac Gym window doesn't appear, ensure you have proper GPU drivers and Isaac Gym installation
- If Quest 3 connection fails, verify USB connection and run `adb devices` to confirm device recognition
- If hand tracking is jittery, adjust IK parameters (`ik_dt`, `ik_thresh`) for smoother motion
- If collision detection isn't working, check that collision groups are properly configured in the simulation

## 🙏 Acknowledgement

We thank the authors of [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision) for sharing their codebase, which provided the foundation for VR-based teleoperation in this project.



