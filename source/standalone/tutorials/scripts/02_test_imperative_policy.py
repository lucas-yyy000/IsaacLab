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
from torchvision import transforms

from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)
from imperative_learning.models.lhsf_model import PlannerNet
from imperative_learning.mpc.mpc import setup_model, setup_mpc, setup_simulator


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

class NetParams:
    def __init__(self, **entries):
        self.__dict__.update(entries)

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
    checkpoints_path = "/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/checkpoints"
    run_name = "ff004055"
    checkpoint_name = "epoch_200.pth"

    with open(os.path.join(checkpoints_path, run_name, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_config = config['data']
    img_size = data_config['img_size']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size)
    ])

    model_config = config['model']
    planner_net = PlannerNet(NetParams(**model_config)).to(device)
    checkpoint = torch.load(os.path.join(checkpoints_path, run_name, checkpoint_name), map_location=device)
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
    states[:, 0] += x_offset
    states[:, 1] += y_offset


    #####  TEST POINTS #####
    # start_location = np.array([4.3638, -54.2186, 30.0])
    # goal_location = np.array([50.36758, 65.43091, 30.0])

    # start_location = np.array([4.3638, -54.2186, 20.0])
    # goal_location_1 = np.array([-80.36758, -50.43091, 25.0])
    # goal_location_2 = np.array([-65.36758, 70.43091, 25.0])

    # start_location = np.array([4.3638, -45.2186, 20.5])
    # goal_location_1 = np.array([-90.36758, -30.43091, 30.0])
    # goal_location_2 = np.array([-10.36758, 35.43091, 25.0])
    # goal = goal_location_1
    # cur_loc = start_location
    # cur_yaw = np.arctan2(goal_location_1[1] - start_location[1], goal_location_1[0] - start_location[0])
    # cur_att = R.from_euler('z', cur_yaw).as_quat()
    # cur_att = np.asarray([cur_att[-1], cur_att[0], cur_att[1], cur_att[2]])
    ########################
    cur_loc = states[0, :3] 
    cur_att = states[0, 3:7]
    start = states[0, :3]
    goal = states[-1, :3]

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
    trajectory_full = []
    trajectory_mpc = []
    collision_cost_prediction = []
    reference_points = []
    # time.sleep(1.0)
    iterations = 0

    first_goal_reached = False


    ########################          MPC Config         #######################
    mpc_horizon = 10
    model = setup_model(V=3.0)
    mpc = setup_mpc(model, mpc_horizon, np.zeros((mpc_horizon, 3)))
    estimator, simulator = setup_simulator(model)
    ###########################################################################

    # Run simulator
    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())
        
        heading = goal - cur_loc
        if np.linalg.norm(heading[:3]) < 15.0:
            break
            # if first_goal_reached:
            #     break
            # else:
            #     first_goal_reached = True
            #     goal = goal_location_2
            #     heading = goal - cur_loc

        # heading /= np.linalg.norm(heading)
        heading /= 10.0

        heading = torch.from_numpy(heading).unsqueeze(dim=0).to(device).to(torch.float32)
        depth_img = camera.data.output["distance_to_image_plane"].squeeze(dim=3).squeeze(dim=0).cpu().numpy()
        # print("Depth image shape: ", depth_img.shape)
        depth_img = transform(depth_img)
        depth_img /= 100.0
        depth_img = depth_img.unsqueeze(dim=0).to(device)
        attitude = torch.from_numpy(cur_att).unsqueeze(dim=0).to(device).to(torch.float32)
        cur_loc_torch = torch.from_numpy(cur_loc).unsqueeze(dim=0).to(device)
        states = torch.cat((heading, attitude), dim=-1)

        with torch.no_grad():
            student_path_local, collision_prediction = planner_net(states, depth_img)

        student_path_global = student_path_local + cur_loc_torch[:, None, None, :]
        collision_prediction = collision_prediction.squeeze(dim=0).cpu().numpy()
        student_path_global = student_path_global.squeeze(dim=0).squeeze(dim=0).cpu().numpy()


        ###########################################         MPC Tracking            ###########################################
        # t = torch.linspace(0, 1, 11)
        # coeffs = natural_cubic_spline_coeffs(t, torch.from_numpy(np.vstack((cur_loc, student_path_global))))
        # spline = NaturalCubicSpline(coeffs)
        # reference_trajectory_interpolated = []
        # for i in range(0, 100):
        #     p = spline.evaluate(torch.tensor((1.0/100.0)*i))
        #     reference_trajectory_interpolated.append(p.cpu().numpy())
        # # print("Current loc and next loc: ", cur_loc, student_path_global[-1])
        # reference_trajectory_interpolated = np.asarray(reference_trajectory_interpolated)
        # # print("Interpolated: ", reference_trajectory_interpolated[:20])
        # # diff = reference_trajectory_interpolated[1:] - reference_trajectory_interpolated[:-1]
        # # print("Diff: ", np.linalg.norm(diff, axis=1))

        # x0 = np.asarray([cur_loc[0], cur_loc[1], cur_loc[2], cur_yaw])
        # mpc.x0 = x0
        # simulator.x0 = x0
        # estimator.x0 = x0
        # mpc.set_initial_guess()
        # mpc.settings.supress_ipopt_output()

        # tracked_trajectory = []
        # for mpc_step in range(10):
        #     tvp_template = mpc.get_tvp_template()
        #     def tvp_fun(t_now):
        #         for k in range(0, 10 + 1):
        #             tvp_template['_tvp',k,'x_ref'] = reference_trajectory_interpolated[mpc_step + k, 0]
        #             tvp_template['_tvp',k,'y_ref'] = reference_trajectory_interpolated[mpc_step + k, 1]
        #             tvp_template['_tvp',k,'z_ref'] = reference_trajectory_interpolated[mpc_step + k, 2]

        #         return tvp_template

        #     mpc.set_tvp_fun(tvp_fun)
        #     mpc.setup()

        #     u0 = mpc.make_step(x0)
        #     y_next = simulator.make_step(u0)
        #     x0 = estimator.make_step(y_next)
        #     tracked_trajectory.append(x0)

        #     # inputs.append(u0)
        #     # print("Control: ", mpc.data['_u'])

        # tracked_trajectory = np.asarray(tracked_trajectory)
        # next_loc = tracked_trajectory[-1, :3].squeeze()
        # next_yaw = tracked_trajectory[-1, -1][0]
        # next_att = R.from_euler('z', next_yaw).as_quat()
        # # print("Next att: ", next_att, next_att.shape)
        # trajectory_mpc.append(tracked_trajectory)
        ####################################################################################################################


        next_loc = student_path_global[-1, :3]
        next_loc_prev = student_path_global[-2, :3]
        next_yaw = np.arctan2(next_loc[1] - next_loc_prev[1], next_loc[0] - next_loc_prev[0])
        next_att = R.from_euler('z', next_yaw).as_quat()
        next_att = np.asarray([next_att[-1], next_att[0], next_att[1], next_att[2]])


        print("Current location: ", cur_loc)
        print("Goal location: ", goal)
        # print("Reference location: ", reference_point)
        print("Next location: ", student_path_global[0, :3])
        # print("Tracked States: ", tracked_trajectory[-1, :3].squeeze())
        # print("Collision probability: ", collision_prediction)

        next_loc_isaac = Rot.apply(next_loc)
        next_att_isaac = Rot*R.from_quat([next_att[1], next_att[2], next_att[3], next_att[0]])
        next_att_isaac = next_att_isaac.as_quat()
        next_att_isaac = np.asarray([next_att_isaac[-1], next_att_isaac[0], next_att_isaac[1], next_att_isaac[2]])
        
        camera_positions = torch.tensor([next_loc_isaac], device=sim.device, dtype=torch.float32)
        camera_orientations = torch.tensor([next_att_isaac], device=sim.device, dtype=torch.float32)

        camera.set_world_poses(camera_positions, camera_orientations, convention='world')

        trajectory.append(cur_loc)
        trajectory_full.append(student_path_global)
        collision_cost_prediction.append(collision_prediction)

        cur_loc = next_loc
        cur_yaw = next_yaw
        cur_att  = next_att
        iterations += 1
        if iterations > 5:
            trajectory_np = np.asarray(trajectory)
            trajectory_full_np = np.asarray(trajectory_full)
            trajectory_mpc_np = np.asarray(trajectory_mpc)
            collision_np = np.asarray(collision_cost_prediction)
            # reference_points_np = np.asarray(reference_points)
            np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/imperative_trajectory.npy", trajectory_np)
            np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/trajectory_full.npy", trajectory_full_np)
            np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/trajectory_mpc.npy", trajectory_mpc_np)
            np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/imperative_trajectory_collision.npy", collision_np)
            # np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/reference_points.npy", reference_points_np)
        
        # time.sleep(0.1)

    trajectory = np.asarray(trajectory)
    trajectory_full_np = np.asarray(trajectory_full)
    trajectory_mpc_np = np.asarray(trajectory_mpc)
    collision_np = np.asarray(collision_cost_prediction)
    # reference_points_np = np.asarray(reference_points)
    np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/imperative_trajectory.npy", trajectory)
    np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/trajectory_full.npy", trajectory_full_np)
    np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/trajectory_mpc.npy", trajectory_mpc_np)
    np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/imperative_trajectory_collision.npy", collision_np)
    # np.save("/home/lucas/Workspace/LowAltitudeFlight/FlightDev/low_altitude_flight/imperative_planning/reference_points.npy", reference_points_np)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
