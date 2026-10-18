import metaworld
import random
import os
import time
import numpy as np
from metaworld.policies import (SawyerBinPickingV2Policy, SawyerBoxCloseV2Policy,
                                SawyerDoorLockV2Policy, SawyerDoorUnlockV2Policy,
                                SawyerHandInsertV2Policy)
from metaworld.policies.play_policy import PlayPolicy
from metaworld.policies.registry import EXPERT_POLICIES
from metaworld.data.dataset import MWDatasetWriter, MWVideoWriter, qlearning_dataset
from datetime import datetime
import gym
import argparse

# Suppress float conversion warnings
gym.logger.set_level(40)

play_cases = [
    # name, policy
    ['bin-picking-v2', PlayPolicy(SawyerBinPickingV2Policy)],
    ['box-close-v2', PlayPolicy(SawyerBoxCloseV2Policy)],
    ['door-lock-v2', PlayPolicy(SawyerDoorLockV2Policy)],
    ['door-unlock-v2', PlayPolicy(SawyerDoorUnlockV2Policy)],
    ['hand-insert-v2', PlayPolicy(SawyerHandInsertV2Policy)]
]


def gen_data(tasks, num_traj, noise, res, include_depth, camera, data_dir_path,
             write_data=True, write_video=False, video_fps=80,
             use_play_policy=False, counter_max=None, grip_flip_p=None):
    res = (res, res)
    MAX_steps_at_goal = 10
    act_tolerance = 1e-5
    lim = 1 - act_tolerance

    cases = play_cases if use_play_policy else list(EXPERT_POLICIES.items())

    for case in cases:

        if case[0] not in tasks:
            continue

        task_name = case[0]
        policy = case[1]

        if use_play_policy:
            policy.counter_max = counter_max
            policy.grip_flip_p = grip_flip_p

        print(f'----------Running task {task_name}------------')

        # Note that, although the environment will generate dense reward (goal_cost_reward=False), we will be able to construct any goal-cost reward and subgoal reward
        # when we load this dataset.
        env = metaworld.mw_gym_make(task_name, goal_cost_reward=False, stop_at_goal=True, steps_at_goal=MAX_steps_at_goal, cam_height=res[0], cam_width=res[1], depth=include_depth, train_distrib=True)
        action_space_ptp = env.action_space.high - env.action_space.low

        num_successes = 0
        dt = datetime.now()
        height, width = res

        data_file_name = task_name + '-num-traj_' + str(num_traj) + '-noise_' + str(noise) + (("-PLAYPOLICY"+"-ctr-max_" +str(counter_max) + "-grip-flip-p_" + str(grip_flip_p)) if use_play_policy else "") + '-res_' + str(height) + '_' + str(width) + '-cam_' + camera + '-depth_' + str(include_depth) + '_' + dt.strftime("%d-%m-%Y-%H.%M.%S") + '.hdf5'
        video_path_root = 'movies'
        video_dir_path = os.path.join(video_path_root, task_name + '-noise_' + str(noise) + (("-PLAYPOLICY"+"-ctr-max_" +str(counter_max) + "-grip-flip-p_" + str(grip_flip_p)) if use_play_policy else "") + '-res_' + str(height) + '_' + str(width) + '-cam_' + camera + '_' + dt.strftime("%d-%m-%Y-%H.%M.%S"))
        task_noise_dir_path = os.path.join(data_dir_path,task_name,'Sawyer', f'noise{noise}')
        assert not os.path.exists(task_noise_dir_path), f"ERROR: {task_noise_dir_path} already exists. You are risking generating more data than you intend to without noticing."
        os.makedirs(task_noise_dir_path)
        data_writer = MWDatasetWriter(task_noise_dir_path, data_file_name, env, task_name, res, camera, include_depth, act_tolerance, MAX_steps_at_goal, write_data=write_data)

        for attempt in range(num_traj):
            video_writer = MWVideoWriter(video_dir_path, task_name + '-' + str(attempt + 1), video_fps, res, write_video=write_video)

            episode = []
            stages = []

            if use_play_policy:
                policy.reset()

            state = env.reset()
            start_time = time.time()

            for t in range(env.max_path_length):
                action = policy.get_action(state['full_state'])
                action = np.random.normal(action, noise * action_space_ptp)
                # Clip the action
                action = np.clip(action, -lim, lim)
                new_state, reward, done, info = env.step(action)
                episode.append((state['full_state'], state['proprio_state'], state['image'], state['depth'], action, reward, done, info))

                if use_play_policy:
                    stages.append(policy.current_stage)

                state = new_state

                if done:
                    if info['task_accomplished']:
                        print(f'Attempt {attempt + 1} succeeded at step {t}')
                        num_successes += 1
                        end_time = time.time()
                    else:
                        print(f'Attempt {attempt + 1} ended unsuccessfully at time step {t}')
                        end_time = time.time()

                    print(f"Average time per step: {(end_time-start_time) / t}")
                    break

            if use_play_policy:
                if 'play' in stages:
                    # Find which stage preceded play
                    prev_stage = stages[stages.index('play')-1]

                    # Find indices of that stage
                    prev_stage_indices = [i for i, v in enumerate(stages) if v == prev_stage]
                    # Pick one randomly
                    start_index = random.choice(prev_stage_indices)
                    print(f'Starting recording from t = {start_index}')
                else:
                    start_index = None
            else:
                start_index = 0

            if start_index is not None:
                # Write data
                for tup in episode[start_index:]:
                    image = tup[2]
                    data_writer.append_data(*tup)
                    video_writer.write(image)

                data_writer.write_trajectory()
            else:
                print('Skipping trajectory (play did not kick in)')

        data_writer.close()
        print('--------------------------------------------------------\n')
        print(f'Success rate for {task_name}: {num_successes / num_traj}\n')

        # Check the created dataset
        if write_data:
            qlearning_dataset(os.path.join(task_noise_dir_path, data_file_name), reward_type='subgoal')



def add_boolean_arg(parser, name, true, false, default):
    assert true.startswith('--') and false.startswith('--')
    assert type(default) is bool
    true_false = parser.add_mutually_exclusive_group()
    true_false.add_argument(true, dest=name, action='store_true')
    true_false.add_argument(false, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


## Example command for generating data with a noisy expert policy, including a depth camera stream:
#python metaworld/data/training_data_gen.py --tasks=door-open-v2  -d=data --num_traj=10 --noise=0.1 --res=84 -f=20 --camera=corner --write_data --include_depth --nowrite_video

## Example command for generating data with a play policy, with recording a video for visualization:
#python metaworld/data/training_data_gen.py --tasks=door-lock-v2 -d=data --num_traj=50 --noise=0 --res=84 -f=20 --use_play_policy --grip_flip_p=0.01 --camera=corner --write_data --write_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasks", type=str, help = "Tasks for which to generate trajectories from scripted policies")
    parser.add_argument("-n", "--num_traj", type=int, help = "Number of trajectories to generate for each task")
    parser.add_argument("-p", "--noise", type=float, default=0, help = "Action noise as a fraction of the action space, e.g., 0.1")
    parser.add_argument("-r", "--res", type=int, default=84, help = "Resolution of image observations (r x r)")
    parser.add_argument("-c", "--camera", type=str, default='corner', help = "Camera. Possible values: 'corner', 'topview', 'corner2', 'corner3', 'behindGripper', 'gripperPOV'")
    parser.add_argument("-f", "--video_fps", type=int, default=80, help = "Fps for recording videos. Ignored if the --nowrite_video flag is present.")
    parser.add_argument("-d", "--data_dir_path", type=str, default='data', help = "Directory where the demonstration data is to be written. Ignored if the ---nowrite_data flag is present.")
    # Should we generate depth frames (HxW arrays whose entries are distances from the camera to objects in the scene, *in millimeters*) in addition to RGB frames?
    add_boolean_arg(parser, 'include_depth', true='--include_depth', false='--noinclude_depth', default=False)
    add_boolean_arg(parser, 'write_data', true='--write_data', false='--nowrite_data', default=True)
    add_boolean_arg(parser, 'write_video', true='--write_video', false='--nowrite_video', default=False)
    add_boolean_arg(parser, 'use_play_policy', true='--use_play_policy', false='--use_expert_policy', default=False)
    # args below only for play policy
    parser.add_argument('--counter_max', type=int, default=100)
    parser.add_argument('--grip_flip_p', type=float, default=0.005)
    args = parser.parse_args()

    tasks = args.tasks.split(',')

    print('\n')
    print(f'Generating {args.num_traj} trajectories with action noise {args.noise} for tasks {tasks} with video resolution {args.res}x{args.res} and {args.camera} camera view.')
    if args.write_video:
        print(f'Videos will be generated at {args.video_fps} fps\n')

    gen_data(tasks, args.num_traj, args.noise, args.res, args.include_depth, args.camera, args.data_dir_path,
             write_data=args.write_data, write_video=args.write_video, video_fps=args.video_fps,
             use_play_policy=args.use_play_policy, counter_max=args.counter_max, grip_flip_p=args.grip_flip_p)
