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
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1e-3)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0, 0, 50.0], [100.0, 100.0, 30.0])

    action_registry = omni.kit.actions.core.get_action_registry()
    # switches to camera lighting
    action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
    action.execute()

    prim_utils.create_prim("/World/Origin_00", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensorLevel",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane"
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera = Camera(cfg=camera_cfg)

    camera_tilted_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensorTilted",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane"
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    camera_tilted = Camera(cfg=camera_tilted_cfg)
    pitch_30 = R.from_euler('y', 30, degrees=True)
    # design the scene
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    camera_path = "/World/Origin_00/CameraSensorTilted"
    viewport = get_active_viewport()
    if not viewport:
        raise RuntimeError("No active Viewport")
    # Set the Viewport's active camera to the
    # camera prim path you want to switch to.
    viewport.camera_path = camera_path


    # Create replicator writer
    level_output_dir = os.path.join("/home/lucas/Workspace/LowAltitudeFlight/TrainingData", "img_data_level")
    level_rep_writer = rep.BasicWriter(
        output_dir=level_output_dir,
        frame_padding=0
    )

    tilted_output_dir = os.path.join("/home/lucas/Workspace/LowAltitudeFlight/TrainingData", "img_data_tilted")
    tilted_rep_writer = rep.BasicWriter(
        output_dir=tilted_output_dir,
        frame_padding=0
    )


    base_path = "/home/lucas/Workspace/LowAltitudeFlight/TrainingData/trajectories"
    x_offset = -114.6951789855957
    y_offset = -78.78693771362305
    Rot = R.from_matrix(np.asarray([
        [0, -1.0, 0],
        [1.0, 0, 0],
        [0, 0, 1]
    ]))
    
    num=0
    states = genfromtxt(os.path.join(base_path, f"{num}_states.csv"), delimiter=',')
    states = states[1:, :-1]
    states[:, 0] += x_offset
    states[:, 1] += y_offset
    step_idx = 0
    state = states[step_idx, :]
    step_max = len(states)
    pos = Rot.apply(state[:3])
    attitude_start_idx = 6
    att = Rot*R.from_quat([state[attitude_start_idx+1], state[attitude_start_idx+2], state[attitude_start_idx+3], state[attitude_start_idx]])
    att_tilted = att*pitch_30
    att = att.as_quat()
    att_tilted = att_tilted.as_quat()
    att = np.asarray([att[-1], att[0], att[1], att[2]])
    att_tilted = np.asarray([att_tilted[-1], att_tilted[0], att_tilted[1], att_tilted[2]])
    camera_positions = torch.tensor([pos], device=sim.device, dtype=torch.float32)
    camera_orientations = torch.tensor([att], device=sim.device, dtype=torch.float32)
    camera_tilted_orientations = torch.tensor([att_tilted], device=sim.device, dtype=torch.float32)

    camera.set_world_poses(camera_positions, camera_orientations, convention='world')
    camera_tilted.set_world_poses(camera_positions, camera_tilted_orientations, convention='world')
    print("Set camera pose successfully.")
    # camera_index = 0

    # time.sleep(5.0)
    # Run simulator
    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())
        camera_tilted.update(dt=sim.get_physics_dt())

        # Print camera info
        print(camera)
        print(camera_tilted)
        # if "rgb" in camera.data.output.keys():
        #     print("Received shape of rgb image  : ", camera.data.output["rgb"].shape)
        # if "distance_to_image_plane" in camera.data.output.keys():
        #     print("Received shape of depth image : ", camera.data.output["distance_to_image_plane"].shape)

        # Save images from camera at camera_index
        # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        level_cam_data = convert_dict_to_backend(
            {k: v[0] for k, v in camera.data.output.items()}, backend="numpy"
        )
        tilted_cam_data = convert_dict_to_backend(
            {k: v[0] for k, v in camera_tilted.data.output.items()}, backend="numpy"
        )
        # Extract the other information
        level_cam_info = camera.data.info[0]
        tilted_cam_info = camera_tilted.data.info[0]
        print("Traj: ", num, " Idx: ", step_idx)
        # Pack data back into replicator format to save them using its writer
        rep_output_level = {"annotators": {}}
        for key, data, info in zip(level_cam_data.keys(), level_cam_data.values(), level_cam_info.values()):
            # print('Key Data Info: ', key, data, info)
            if info is not None:
                rep_output_level["annotators"][key] = {"render_product": {"data": data, **info}}
            else:
                print("<<====================   No INFO   ====================>>")
                # print("Key: ", key)
                # key_indexed = key + f"_{num}_{step_idx}"
                rep_output_level["annotators"][key] = {"render_product": {"data": data}}
        rep_output_tilted = {"annotators": {}}
        for key, data, info in zip(tilted_cam_data.keys(), tilted_cam_data.values(), tilted_cam_info.values()):
            # print('Key Data Info: ', key, data, info)
            if info is not None:
                rep_output_tilted["annotators"][key] = {"render_product": {"data": data, **info}}
            else:
                print("<<====================   No INFO   ====================>>")
                # print("Key: ", key)
                # key_indexed = key + f"_{num}_{step_idx}"
                rep_output_tilted["annotators"][key] = {"render_product": {"data": data}}
        
        # Save images
        # Note: We need to provide On-time data for Replicator to save the images.
        rep_output_level["trigger_outputs"] = {"on_time": camera.frame[0]}
        level_rep_writer.write(rep_output_level)

        rep_output_tilted["trigger_outputs"] = {"on_time": camera_tilted.frame[0]}
        tilted_rep_writer.write(rep_output_tilted)

        step_idx += 1
        if step_idx >= step_max:
            num += 1
            states = genfromtxt(os.path.join(base_path, f"{num}_states.csv"), delimiter=',')
            states = states[1:, :-1]
            states[:, 0] += x_offset
            states[:, 1] += y_offset
            step_idx = 0
            step_max = len(states)
            break
        
        state = states[step_idx, :]
        pos = Rot.apply(state[:3])
        att =  Rot*R.from_quat([state[attitude_start_idx+1], state[attitude_start_idx+2], state[attitude_start_idx+3], state[attitude_start_idx]])
        att_tilted = att*pitch_30
        att = att.as_quat()
        att_tilted = att_tilted.as_quat()
        att = np.asarray([att[-1], att[0], att[1], att[2]])
        att_tilted = np.asarray([att_tilted[-1], att_tilted[0], att_tilted[1], att_tilted[2]])
        camera_positions = torch.tensor([pos], device=sim.device, dtype=torch.float32)
        camera_orientations = torch.tensor([att], device=sim.device, dtype=torch.float32)
        camera_tilted_orientations = torch.tensor([att_tilted], device=sim.device, dtype=torch.float32)

        camera.set_world_poses(camera_positions, camera_orientations, convention='world')
        camera_tilted.set_world_poses(camera_positions, camera_tilted_orientations, convention='world')


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
