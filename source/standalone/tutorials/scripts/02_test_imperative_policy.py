# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)
from imperative_learning.models.baseline import ForwardDepthModel



# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy import genfromtxt
import yaml
import os
import random
import time
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep

import omni

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend
import omni.kit.actions.core
from omni.kit.viewport.utility import get_active_viewport



# torch.set_default_dtype(torch.float32)

def main():
    """Main function."""
    omni.usd.get_context().open_stage("/home/lucas/Documents/Omniverse/IsaacSimTerrains/terrains/terrain_0.usd")
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0, 0, 50.0], [100.0, 100.0, 30.0])

    action_registry = omni.kit.actions.core.get_action_registry()
    # switches to camera lighting
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
    action.execute()

    prim_utils.create_prim("/World/Origin_00", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane"
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg)

    # design the scene
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")


    camera_path = "/World/Origin_00/CameraSensor"
    viewport = get_active_viewport()
    if not viewport:
        raise RuntimeError("No active Viewport")
    # Set the Viewport's active camera to the
    # camera prim path you want to switch to.
    viewport.camera_path = camera_path

    # Load Student Policy
    device='cuda'
    root_path = "/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning"

    run_name = "6eb69668"
    checkpoint_path = os.path.join(root_path, "checkpoints", run_name)
    checkpoint_name = 'epoch_70.pth'

    planner_net = ForwardDepthModel([256, 256], hidden_dim=512).to(device)
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)
    planner_net.load_state_dict(checkpoint['model_state_dict'])
    planner_net.eval()

    base_path = "/home/lucas/Workspace/LowAltitudeFlight/expert_demonstrations/ExpertDemonstrations"
    x_offset = -114.6951789855957
    y_offset = -78.78693771362305
    Rot = R.from_matrix(np.asarray([
        [0, -1.0, 0],
        [1.0, 0, 0],
        [0, 0, 1]
    ]))
    
    num=2
    states = genfromtxt(os.path.join(base_path, f"{num}_states.csv"), delimiter=',')
    states = states[1:, :-1]

    cur_loc = states[0, :3] + np.asarray([x_offset, y_offset, 0])
    cur_att = states[0, 3:7]
    goal = states[-1, :3] + np.asarray([x_offset, y_offset, 0])

    cur_loc_isaac = Rot.apply(cur_loc)
    cur_att_isaac = Rot*R.from_quat([cur_att[1], cur_att[2], cur_att[3], cur_att[0]])
    cur_att_isaac = cur_att_isaac.as_quat()
    cur_att_isaac = np.asarray([cur_att_isaac[-1], cur_att_isaac[0], cur_att_isaac[1], cur_att_isaac[2]])

    camera_positions = torch.tensor([cur_loc_isaac], device=sim.device, dtype=torch.float32)
    camera_orientations = torch.tensor([cur_att_isaac], device=sim.device, dtype=torch.float32)
    camera.set_world_poses(camera_positions, camera_orientations, convention='world')
    print("Set camera pose successfully.")
    print("Starting at: ", cur_loc)
    trajectory = []
    time.sleep(1.0)
    iterations = 0
    # Run simulator
    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())

        # heading = (goal - cur_loc) / 100.0
        heading = goal - cur_loc

        if np.linalg.norm(heading[:2]) < 20.0:
            break

        heading /= np.linalg.norm(heading)
        heading = torch.from_numpy(heading).unsqueeze(dim=0).to(device).to(torch.float32)
        depth_img = camera.data.output["distance_to_image_plane"].squeeze(dim=3).to(device)
        depth_img /= 1000.0
        attitude = torch.from_numpy(cur_att).unsqueeze(dim=0).to(device).to(torch.float32)
        # print("Attitude shape: ", attitude, attitude.shape)
        cur_loc_torch = torch.from_numpy(cur_loc).unsqueeze(dim=0).to(device)
        with torch.no_grad():
            planned_states = planner_net(depth_img, heading, attitude, cur_loc_torch)

        # planned_states = planned_states.squeeze(dim=0).cpu().numpy()
        batch_size, num_p, _ = planned_states.shape
        t = torch.linspace(0, 1, num_p+1).to(planned_states.device)
        coeffs = natural_cubic_spline_coeffs(t, torch.cat((cur_loc_torch.unsqueeze(dim=1), planned_states), dim=1))
        spline = NaturalCubicSpline(coeffs)
        t = torch.linspace(0, 1, 20).to(planned_states.device)
        trajectory_interpolated = spline.evaluate(t)
        trajectory_interpolated = trajectory_interpolated.squeeze(dim=0).cpu().numpy()

        
        next_loc = trajectory_interpolated[3, :3]
        yaw = np.arctan2(next_loc[1] - cur_loc[1], next_loc[0] - cur_loc[0])
        next_att = R.from_euler('z', yaw).as_quat()
        next_att = np.asarray([next_att[-1], next_att[1], next_att[2], next_att[3]])

        print("Current location: ", cur_loc)
        print("Goal location: ", goal)
        print("Planned states", planned_states[:, :3])

        next_loc_isaac = Rot.apply(next_loc)
        next_att_isaac = Rot*R.from_quat([next_att[1], next_att[2], next_att[3], next_att[0]])
        next_att_isaac = next_att_isaac.as_quat()
        next_att_isaac = np.asarray([next_att_isaac[-1], next_att_isaac[0], next_att_isaac[1], next_att_isaac[2]])
        
        camera_positions = torch.tensor([next_loc_isaac], device=sim.device, dtype=torch.float32)
        camera_orientations = torch.tensor([next_att_isaac], device=sim.device, dtype=torch.float32)

        camera.set_world_poses(camera_positions, camera_orientations, convention='world')

        trajectory.append(cur_loc)
        cur_loc = next_loc
        cur_att  = next_att
        iterations += 1
        if iterations > 5:
            trajectory_np = np.asarray(trajectory)
            np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/imperative_trajectory.npy", trajectory_np)
        
        time.sleep(0.05)

    trajectory = np.asarray(trajectory)
    np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/imperative_trajectory.npy", trajectory)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
