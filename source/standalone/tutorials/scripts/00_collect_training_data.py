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

# torch.set_default_dtype(torch.float32)

def main():
    """Main function."""
    omni.usd.get_context().open_stage("/home/lucas/Workspace/IsaacSimTerrains/terrains/terrain_0_world.usd")
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


    # Create replicator writer
    output_dir = os.path.join("/media/lucas/T9/TrainingData", "data")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0
    )


    base_path = "/media/lucas/T9/ExpertDemonstrations/"
    x_offset = -114.6951789855957
    y_offset = -78.78693771362305
    Rot = R.from_matrix(np.asarray([
        [0, -1.0, 0],
        [1.0, 0, 0],
        [0, 0, 1]
    ]))
    
    num=0
    states = genfromtxt(base_path + f"{num}_states.csv", delimiter=',')
    states = states[1:, :-1]
    states[:, 0] += x_offset
    states[:, 1] += y_offset
    step_idx = 0
    state = states[step_idx, :]
    step_max = len(states)
    pos = Rot.apply(state[:3])
    att = Rot*R.from_quat([state[4], state[5], state[6], state[3]])
    att = att.as_quat()
    att = np.asarray([att[-1], att[0], att[1], att[2]])
    camera_positions = torch.tensor([pos], device=sim.device, dtype=torch.float32)
    camera_orientations = torch.tensor([att], device=sim.device, dtype=torch.float32)
    # camera.set_world_poses_from_view(camera_positions, camera_targets)
    camera.set_world_poses(camera_positions, camera_orientations, convention='world')
    print("Set camera pose successfully.")
    camera_index = 0
    time.sleep(5.0)
    # Run simulator
    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())
                # Print camera info
        print(camera)
        if "rgb" in camera.data.output.keys():
            print("Received shape of rgb image  : ", camera.data.output["rgb"].shape)
        if "distance_to_image_plane" in camera.data.output.keys():
            print("Received shape of depth image : ", camera.data.output["distance_to_image_plane"].shape)
        # Save images from camera at camera_index
        # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )

        # Extract the other information
        single_cam_info = camera.data.info[camera_index]
        print("Traj: ", num, " Idx: ", step_idx)
        # Pack data back into replicator format to save them using its writer
        rep_output = {"annotators": {}}
        for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
            # print('Key Data Info: ', key, data, info)
            if info is not None:
                rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
            else:
                print("<<====================   No INFO   ====================>>")
                # print("Key: ", key)
                # key_indexed = key + f"_{num}_{step_idx}"
                rep_output["annotators"][key] = {"render_product": {"data": data}}
        # Save images
        # Note: We need to provide On-time data for Replicator to save the images.
        rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
        rep_writer.write(rep_output)

        step_idx += 1
        if step_idx >= step_max:
            num += 1
            states = genfromtxt(base_path + f"{num}_states.csv", delimiter=',')
            states = states[1:, :-1]
            states[:, 0] += x_offset
            states[:, 1] += y_offset
            step_idx = 0
            step_max = len(states)
        
        state = states[step_idx, :]
        pos = Rot.apply(state[:3])
        att = Rot*R.from_quat([state[4], state[5], state[6], state[3]])
        att = att.as_quat()
        att = np.asarray([att[-1], att[0], att[1], att[2]])
        camera_positions = torch.tensor([pos], device=sim.device, dtype=torch.float32)
        camera_orientations = torch.tensor([att], device=sim.device, dtype=torch.float32)
        # camera_positions = torch.tensor([np.asarray([-state[1], state[0], state[2]])], device=sim.device, dtype=torch.float32)
        # camera_orientations = torch.tensor([state[3:7]], device=sim.device, dtype=torch.float32)
        # camera.set_world_poses_from_view(camera_positions, camera_targets)
        camera.set_world_poses(camera_positions, camera_orientations, convention='world')


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
