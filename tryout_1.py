import torch
import numpy as np
import sys

sys.path.append("G:\GIT\TD3-PyTorch-BipedalWalker-v2")
from TD3 import TD3
from utils import ReplayBuffer

import time
import matplotlib.pyplot as plt

from labEnv import LabEnv, MobRob
import gc


def main():
    train = True
    vrepHeadlessMode = False
    simulate = True
    plot = True

    ######### Hyperparameters #########
    env_name = "MobRob-Corridor"
    log_interval = 10           # print avg reward after interval
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.1
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    episodes = 2000         # max num of episodes
    steps = 500       # max timesteps in one episode
    directory = "./preTrained/{}".format(env_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    ###################################

    state_dim = 6  # TODO: simulate battery? 6 are the number of proxy sensors    # x, y, vx, vy, v_yaw, prox 0 ... prox5
    action_dim = 2
    min_action = float(-2)
    max_action = float(2)

    desiredState = [-1.4, 0.3, -np.pi, 0.0, 0.0, 0.0]  # x, y, yawAngle, vx, vy, yawVelocity

    if train:
        mobRob = MobRob(['MobRob'],
                        ['leftMotor', 'rightMotor'],
                        ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                         'proximitySensor5'])
        env = LabEnv(mobRob, vrepHeadlessMode)

        policy = TD3(lr, state_dim, action_dim, max_action)

        if policy is not None:
            print('TD3 agent initialized')
        else:
            print('TD3 agent failed to initialize')
            exit(-1)

        replay_buffer = ReplayBuffer()

        # logging variables:
        log_f = open("log.txt", "w+")

        total_num_of_steps = 0
        actions = np.zeros((episodes, steps+1, action_dim), dtype=np.float)
        total_rewards = []
        save_rewards = []
        durations = []
        avg_reward = 0

        for episode in range(episodes):
            state = env.reset(desiredState)
            start_time = time.time()
            reason = ''
            episode_rewards = []
            for step in range(steps+1):
                total_num_of_steps += 1

                # select action and add exploration noise:
                action = policy.select_action(state)
                action = action + np.random.normal(0, exploration_noise, size=action_dim)
                action = action.clip(min_action, max_action)
                actions[episode][step] = action
                # print(action)

                # take action in env:
                next_state, reward, done = env.step(action, desiredState)
                replay_buffer.add((state, action, reward, next_state, float(done)))
                state = next_state

                episode_rewards.append(reward)

                # if episode is done then update policy:
                if done or step == (steps - 1):
                    policy.update(replay_buffer, step, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                    break

                if step < steps and done and ~env.collision:
                    reason = 'COMPLETED'
                    break

                if step == steps - 1: # time budget for episode was overstepped
                    reason = 'TIMEOUT  '
                    break

                if env.collision:
                    reason = 'COLLISION'
                    break

            episode_reward = np.sum(episode_rewards)
            min_score = np.min(episode_rewards)
            max_score = np.max(episode_rewards)
            total_rewards.append(episode_reward)
            duration = time.time() - start_time
            durations.append(duration)
            save_rewards.append([total_rewards[episode], episode])

            eta = np.mean(durations)*(episodes-episode) / 60 / 60
            if eta < 1.0:
                etaString = str(np.round(eta * 60, 2)) + " min"
            else:
                etaString = str(np.round(eta, 2)) + " h"

            # logging updates:
            log_f.write('{},{}\n'.format(episode, episode_reward))
            log_f.flush()

            if len(total_rewards) >= log_interval:
                avg_reward = np.mean(total_rewards[-log_interval:])

            print(
                '\rEpisode {}\t{}\tEpisode reward: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tAvg_reward: {:.2f}\tDuration: {:.2f}\tETA: {}'
                    .format(episode, reason, episode_reward, min_score, max_score, avg_reward, duration, etaString))

            gc.collect()

            # if average reward > 300 then save and stop traning:
            if len(total_rewards) >= log_interval:
                if (np.mean(total_rewards[-log_interval:])) >= 100:
                    print("########## Solved! ###########")
                    name = filename + '_solved'
                    policy.save(directory, name)
                    log_f.close()
                    break

            if episode > 500:
                policy.save(directory, filename)

        np.save('mean_episode_rewards', save_rewards)
    elif simulate:
        random_seed = 0
        n_episodes = 3
        lr = 0.002
        max_timesteps = 2000

        filename = "TD3_{}_{}".format(env_name, random_seed)
        filename += '_solved'
        directory = "./preTrained/{}".format(env_name)

        mobRob = MobRob(['MobRob'],
                        ['leftMotor', 'rightMotor'],
                        ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                         'proximitySensor5'])
        env = LabEnv(mobRob, vrepHeadlessMode)
        policy = TD3(lr, state_dim, action_dim, max_action)

        if policy is not None:
            print('TD3 agent initialized')
        else:
            print('TD3 agent failed to initialize')
            exit(-1)

        policy.load_actor(directory, filename)

        for episode in range(1, n_episodes + 1):
            ep_reward = 0
            state = env.reset(desiredState)
            for step in range(max_timesteps):
                action = policy.select_action(state)
                state, reward, done, _ = env.step(action,desiredState)
                ep_reward += reward

                if done:
                    break

            print('Episode: {}\tReward: {}'.format(episode, int(ep_reward)))
            ep_reward = 0

    elif plot:
        mean_episode_rewards = np.load('./mean_episode_rewards.npy')

        f, ax = plt.subplots(1, sharex=True, sharey=False)

        # # Always
        ax.set_ylabel("Mean episode rewards")
        ax.set_xlabel("Number of trials")

        # Highlight the starting x axis
        ax.axhline(0, color="#AAAAAA")
        ax.plot(mean_episode_rewards[:, 1], mean_episode_rewards[:, 0])
        ax.grid(True)

        plt.show()

if __name__ == "__main__":
    main()