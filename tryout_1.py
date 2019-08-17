import torch
import numpy as np
import sys, os

sys.path.append("C:/visageGIT\TD3-PyTorch-BipedalWalker-v2")
from SAC import SAC
from utils import ReplayBuffer, EnvironmentTiles

import time
import matplotlib.pyplot as plt

from labEnv import LabEnv, MobRob
import gc


def main():
    train = True
    vrepHeadlessMode = False
    simulate = True
    plot = True

    envLength = 2.5
    envWidth = 2.0
    envOriginCoordinate = [-0.5, -1.0]  # double T maze
    tileSize = [0.1, 0.1]

    ######### Hyperparameters #########
    # env_name = "MobRob-Corridor"
    env_name = "MobRob-U_maze"
    # env_name = "MountainCar-v0"
    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    min_batch_size = 10000
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    initial_exploration_noise = 0.5     # was 0.1
    exploration_noise = 0.25
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    episodes = 10000            # max num of episodes
    steps = 1000                # max timesteps in one episode
    iterations_per_step = 1
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################

    if not os.path.isdir(directory):
        os.makedirs(directory)

    state_dim =12  # TODO: simulate battery? 6 are the number of proxy sensors    # x, y, vx, vy, v_yaw, prox 0 ... prox5
    action_dim = 2
    min_action = float(-2)
    max_action = float(2)
    actionBounds = [min_action, max_action]

    # terminalState = [1.65, 1.8, 0.5, 0.85]  # xMin, xMax, yMin, yMax  U_maze
    # terminalState = [-0.35, -0.15, 0.5, 0.85]  # xMin, xMax, yMin, yMax    double_T_maze
    terminalState = [0.0, 0.5, -0.6, -0.85]  # xMin, xMax, yMin, yMax    T_maze



    if train:
        environmentTiles = EnvironmentTiles(envWidth, envLength, envOriginCoordinate, tileSize)

        mobRob = MobRob(['MobRob'],
                        ['leftMotor', 'rightMotor'],
                        ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                         'proximitySensor5'])

        env = LabEnv(mobRob, terminalState, actionBounds, environmentTiles, vrepHeadlessMode)

        policy = SAC(lr, state_dim, action_dim, max_action)

        if policy is not None:
            print('TD3 agent initialized')
        else:
            print('TD3 agent failed to initialize')
            exit(-1)

        # logging variables:
        log_f = open("log.txt", "w+")

        total_num_of_steps = 0
        actions = np.zeros((episodes, steps+1, action_dim), dtype=np.float)
        total_rewards = []
        save_rewards = []
        durations = []
        avg_reward = 0

        for episode in range(episodes):
            passed, state = env.reset()
            start_time = time.time()
            reason = ''
            episode_rewards = []
            finished_episode = False
            log_f.write('Episode {}\n'.format(episode))
            environmentTiles.reset()
            for step in range(steps+1):
                total_num_of_steps += 1

                # select action and add exploration noise:
                action = policy.select_action(state)

                # take action in env:
                next_state, reward, done, passed = env.step(action)
                if not passed:
                    reason = 'FAILED'
                    break

                policy.store(state, action, reward, next_state, float(done))

                if policy.num_transition >= policy.min_buffer_size:
                    policy.update()

                # log_f.write('  Step: {} vx: {} vy: {} Resultant speed: {} yawRate: {} Step reward: {}\n'.format(step, state[0], state[1], np.linalg.norm(state[:2]), state[2], reward))
                state = next_state

                episode_rewards.append(reward)

                if step <= steps and done and ~env.collision:
                    reason = 'COMPLETED'
                    finished_episode = True

                if step == steps:
                    reason = 'TIMEOUT  '
                    finished_episode = True

                if env.collision:
                    reason = 'COLLISION'
                    finished_episode = True

                if finished_episode:
                    break

            if reason != 'FAILED':
                episode_reward = np.sum(episode_rewards)
                min_score = np.min(episode_rewards)
                max_score = np.max(episode_rewards)
                total_rewards.append(episode_reward)
                duration = time.time() - start_time
                durations.append(duration)
                save_rewards.append([episode_reward, episode])

                eta = np.mean(durations)*(episodes-episode) / 60 / 60
                if eta < 1.0:
                    etaString = str(np.round(eta * 60, 2)) + " min"
                else:
                    etaString = str(np.round(eta, 2)) + " h"

                # logging updates:
                log_f.write('  Episode reward: {}\n'.format(episode_reward))
                log_f.flush()

                if len(total_rewards) >= log_interval:
                    avg_reward = np.mean(total_rewards[-log_interval:])

                print(
                    '\rEpisode {}\t{}\tEpisode reward: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tAvg_reward: {:.2f}\tDuration: {:.2f}\tETA: {}'
                        .format(episode, reason, episode_reward, min_score, max_score, avg_reward, duration, etaString))
            else:
                print('FAILED EPISODE!')
                episodes += 1

            gc.collect()

            # if average reward > 300 then save and stop traning:
            if len(total_rewards) >= log_interval:
                if avg_reward >= 1100:
                    print("########## Solved! ###########")
                    name = filename + '_solved'
                    policy.save()
                    log_f.close()
                    break

            if episode > 100 and episode % log_interval == 0:
                policy.save()

        np.save('mean_episode_rewards', save_rewards)
    elif simulate:
        random_seed = 0
        n_episodes = 3
        lr = 0.002
        max_timesteps = 2000

        filename = "TD3_{}_{}".format(env_name, random_seed)
        # filename += '_solved'
        directory = "./preTrained/{}".format(env_name)

        environmentTiles = EnvironmentTiles(envWidth, envLength, envOriginCoordinate, tileSize)
        mobRob = MobRob(['MobRob'],
                        ['leftMotor', 'rightMotor'],
                        ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                         'proximitySensor5'])
        env = LabEnv(mobRob, terminalState, actionBounds, environmentTiles, vrepHeadlessMode)
        policy = SAC(lr, state_dim, action_dim, max_action)

        if policy is not None:
            print('TD3 agent initialized')
        else:
            print('TD3 agent failed to initialize')
            exit(-1)

        policy.load_actor(directory, filename)

        for episode in range(1, n_episodes + 1):
            ep_reward = 0
            _, state = env.reset()
            for step in range(max_timesteps):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action)
                ep_reward += reward

                if done:
                    break

            print('Episode: {}\tReward: {}'.format(episode, int(ep_reward)))
            ep_reward = 0
        env.stop()

    elif plot:
        mean_episode_rewards = np.load('./mean_episode_rewards.npy')

        f, ax = plt.subplots(1, sharex=True, sharey=False)

        # # Always
        ax.set_ylabel("Episode reward")
        ax.set_xlabel("Number of episodes")

        # Highlight the starting x axis
        ax.axhline(0, color="#AAAAAA")
        ax.plot(mean_episode_rewards[:, 1], mean_episode_rewards[:, 0])
        ax.grid(True)

        plt.show()

if __name__ == "__main__":
    main()