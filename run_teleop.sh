#!/bin/bash

# Quest3 Teleop System Launch Script
# Based on teleop_run.py command line parameters

set -e

echo "=== Quest3 Teleop System ==="

# Command line parameters configuration
# --isaac_output, -o: Path to isaac gym image output directory
ISAAC_OUTPUT="output/"

# --robot_ik_urdf, -ik_u: Path to pinocchio urdf file for IK computation
ROBOT_IK_URDF="assets/franka_pinocchio/robots/franka_panda.urdf"

# --assets_path, -a_p: Path to assets directory
ASSETS_PATH="assets/"

# --init_mode, -i_m: Initial visualization mode ("rgb", "depth", "mask")
INIT_MODE="rgb"

# --vis_camera, -v_s: Enable pygame visualization window, visualize camera view in simulation
VIS_CAMERA=false

# --vis_issac, -v_s: Enable issac gym simulation view
VIS_ISSAC=true

# --ik_dt: IK solver time step
IK_DT=1e-2

# --ik_thresh: IK convergence threshold
IK_THRESH=1e-6

# --ik_max_iterate_step: Maximum IK iteration steps
IK_MAX_ITERATE_STEP=200

# Create output directory
mkdir -p "$ISAAC_OUTPUT"

# Display configuration
echo "Configuration:"
echo "  Output: $ISAAC_OUTPUT"
echo "  Robot URDF: $ROBOT_IK_URDF"
echo "  Assets: $ASSETS_PATH"
echo "  Initial mode: $INIT_MODE"
echo "  Screen visualization: $VIS_SCREEN"

# Control instructions
echo "Controls:"
echo "  Left hand - Thumb+Index: Start recording"
echo "  Left hand - Thumb+Middle: Stop recording"
echo "  Right hand - Thumb+Index: Depth mode"
echo "  Right hand - Thumb+Middle: Mask mode"
echo "  Right hand - Thumb+Ring: RGB mode"
echo "  Exit: Ctrl+C"

# Launch the program
echo "Starting teleop system..."

# 5秒倒计时
echo -e "${YELLOW}Start in 5 sec...${NC}"
for i in {5..1}; do
    echo -n "$i "
    sleep 1
done
echo ""


python teleop_run.py \
    --isaac_output "$ISAAC_OUTPUT" \
    --robot_ik_urdf "$ROBOT_IK_URDF" \
    --assets_path "$ASSETS_PATH" \
    --init_mode "$INIT_MODE" \
    --vis_camera $VIS_CAMERA \
    --vis_issac $VIS_ISSAC \
    --ik_dt $IK_DT \
    --ik_thresh $IK_THRESH \
    --ik_max_iterate_step $IK_MAX_ITERATE_STEP

echo "Program exited"