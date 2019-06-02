import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from ddpg_agent import Agent
from labEnv import LabEnv, MobRob
import gc


def main():
    train = True
    vrepHeadlessMode = False
    simulate = False
    plot = False

    state_dim = 6  # TODO: simulate battery? 6 are the number of proxy sensors
    action_dim = 2
    action_space = np.array([[-2, 2], [-2, 2]])
    action_lim = [-2.0, 2.0]  # 2 o/sec is the max angular speed of each motor, max. linear velocity is 0.5 m/s

    learn_every = 20  # number of steps after which he network update occurs
    num_learn = 10  # number of network updates done in a row

    trials = 1000
    trial_len = 300

    desiredState = [-1.4, 0.3, -np.pi, 0.0, 0.0, 0.0]  # x, y, yawAngle, vx, vy, yawVelocity

    if train:
        mobRob = MobRob(['MobRob'],
                        ['leftMotor', 'rightMotor'],
                        ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                         'proximitySensor5'])
        env = LabEnv(mobRob, vrepHeadlessMode)

        random_seed = 7
        mobRob = Agent(state_dim, action_dim, random_seed)

        if mobRob is not None:
            print('mobRob agent initialized')
        else:
            print('mobRob agent failed to initialize')

        total_num_of_steps = 0
        actions = np.zeros((trials, trial_len+1, action_dim), dtype=np.float)
        episode_rewards = []
        total_rewards = []
        save_rewards = []
        for trial in range(trials):
            cur_state = env.restart()
            mobRob.reset()
            start_time = time.time()
            for step in range(trial_len+1):
                total_num_of_steps += 1
                action = mobRob.act(cur_state)[0]
                actions[trial][step] = action
                # print(action)
                new_state, reward, done = env.step(action, desiredState)
                mobRob.step(cur_state, action, reward, new_state, done)

                cur_state = new_state
                episode_rewards.append(reward)

                if step % learn_every == 0:
                    for _ in range(num_learn):
                        mobRob.start_learn()

                mean_score = np.mean(episode_rewards)
                min_score = np.min(episode_rewards)
                max_score = np.max(episode_rewards)
                total_rewards.append(mean_score)
                duration = time.time() - start_time

                if step < trial_len and done and ~env.getCollision():
                    print(
                        '\rEpisode {}\tCOMPLETED\tMean episode reward: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'
                        .format(trial, mean_score, min_score, max_score, duration))
                    break

                if step == trial_len: # time budget for episode was overstepped
                    print(
                        '\rEpisode {}\tTIMEOUT\tMean episode reward: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'
                        .format(trial, mean_score, min_score, max_score, duration))

                if env.getCollision():
                    print(
                        '\rEpisode {}\tCOLLISION\tMean episode reward: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'
                        .format(trial, mean_score, min_score, max_score, duration))

            save_rewards.append([total_rewards[trial], trial])

            gc.collect()

        torch.save(mobRob.actor.state_dict(), './actor.pth')
        torch.save(mobRob.critic.state_dict(), './critic.pth')
        np.save('mean_episode_rewards', save_rewards)
    elif simulate:
        mobRob = MobRob(['MobRob'],
                        ['leftMotor', 'rightMotor'],
                        ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                         'proximitySensor5'])
        env = LabEnv(mobRob, vrepHeadlessMode)

        random_seed = 7
        mobRob = Agent(state_dim, action_dim, random_seed)

        if mobRob is not None:
            print('mobRob agent initialized')
        else:
            print('mobRob agent failed to initialize')

        mobRob.actor.load_state_dict(torch.load('./Trainings/latest/actor.pth'))
        cur_state = env.restart()
        mobRob.reset()

        for step in range(100000):
            # total_num_of_steps += 1
            # action = dqn_agent.act(cur_state, trial)
            action = mobRob.step(cur_state)
            # actions[trial][step] = action
            # print(action)
            new_state, _, done = env.step(action, desiredState)
            cur_state = new_state

    elif plot:
        mean_episode_rewards = np.load('./Trainings/latest/mean_episode_rewards.npy')

        f, ax = plt.subplots(1, sharex=True, sharey=False)

        # # Always
        ax[2].set_ylabel("Mean episode rewards")
        ax[-1].set_xlabel("Number of trials")

        # Highlight the starting x axis
        ax[2].axhline(0, color="#AAAAAA")
        ax[2].plot(mean_episode_rewards[:, 1], mean_episode_rewards[:, 0])
        ax[2].grid(True)

        plt.show()

if __name__ == "__main__":
    main()