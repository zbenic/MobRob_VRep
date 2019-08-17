import torch
import numpy as np
import sys, os

sys.path.append("C:/visageGIT\TD3-PyTorch-BipedalWalker-v2")
from TD3 import TD3
from utils import ReplayBuffer, EnvironmentTiles

import time
import matplotlib.pyplot as plt

from labEnv import LabEnv, MobRob
import gc

import gym


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
    # env_name = "MobRob-U_maze-working"
    env_name = "MountainCarContinuous-v0"
    # env_name = "Pendulum-v0"

    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    min_batch_size = 100
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.0003
    initial_exploration_noise = 0.1     # was 0.1
    exploration_noise = 0.1
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    episodes = 1000             # max num of episodes
    steps = 2000                 # max timesteps in one episode
    iterations_per_step = 1
    directory = "./preTrained/{}".format(env_name)  # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################

    if not os.path.isdir(directory):
        os.makedirs(directory)


    if train:
        env = gym.make(env_name)

        # policy = TD3(lr, env.observation_space.shape[0], env.action_space.shape[0], env.max_action)
        policy = TD3(lr, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])


        if policy is not None:
            print('TD3 agent initialized')
        else:
            print('TD3 agent failed to initialize')
            exit(-1)

        replay_buffer = ReplayBuffer()
        # replay_buffer.load()

        # logging variables:
        log_f = open("log.txt", "w+")

        total_num_of_steps = 0
        total_rewards = []
        save_rewards = []
        durations = []
        avg_reward = 0

        for episode in range(episodes):
            start_time = time.time()
            state = env.reset()
            episode_rewards = []
            log_f.write('Episode {}\n'.format(episode))
            for step in range(steps+1):
                env.render()
                total_num_of_steps += 1

                # select action and add exploration noise:
                action = policy.select_action(state)
                if replay_buffer.size < min_batch_size:
                    action = action + np.random.normal(0, initial_exploration_noise, size=env.action_space.shape[0])
                else:
                    action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
                # action = action.clip(env.min_action, env.max_action)
                action = action.clip(env.action_space.low[0], env.action_space.high[0])

                # take action in env:
                next_state, reward, done, info = env.step(action)

                replay_buffer.add((state, action, reward, next_state, float(done)))
                state = next_state

                episode_rewards.append(reward)

                if done:
                    if replay_buffer.size >= min_batch_size:
                        policy.update(replay_buffer, 10, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    break
                elif step == steps:
                    if replay_buffer.size >= min_batch_size:
                        policy.update(replay_buffer, 10, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)


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
                '\rEpisode {}\tEpisode reward: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tAvg_reward: {:.2f}\tDuration: {:.2f}\tETA: {}'
                    .format(episode, episode_reward, min_score, max_score, avg_reward, duration, etaString))

            gc.collect()

            # if average reward > 300 then save and stop traning:
            if len(total_rewards) >= log_interval:
                if avg_reward >= 1100:
                    print("########## Solved! ###########")
                    name = filename + '_solved'
                    policy.save(directory, name)
                    replay_buffer.save()
                    log_f.close()
                    break

            if episode > 500 and episode % log_interval == 0:
                policy.save(directory, filename)
                replay_buffer.save()

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
        policy = TD3(lr, state_dim, action_dim, max_action)

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