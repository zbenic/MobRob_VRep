from __future__ import division
import numpy as np

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import random
import matplotlib.pyplot as plt
import time
from collections import deque
import gc

from SumTree import SumTree

import subprocess
import time
import array
from PIL import Image
from numpy.linalg import norm
import sys

try:
    import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')


EPS = 0.003

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Torch will use " + device.__str__() + " as device.")


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.min()
        self.high = action_space.max()
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)



class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, action_lim, learning_rate=3e-4):
        super(Actor, self).__init__()
        self.action_lim = action_lim
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x * torch.from_numpy(np.array(self.action_lim)).float()


class DQN:
    def __init__(self, env, state_dim, action_dim, action_lim, gamma=0.85, epsilon=1.0, tau=0.125, learning_rate=0.001, hidden_dim=256, batch_size=128, max_memory_size=100000):
        self.env = env
        self.batch_size = batch_size
        self.memory = Memory(max_memory_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.tau = tau

        self.input_num = state_dim
        self.output_num = action_dim
        self.action_lim = action_lim
        self.hidden_dim = hidden_dim

        self.actor = Actor(state_dim, hidden_dim, action_dim, action_lim)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_lim)
        hard_update(self.actor_target, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate/10)

        self.critic = Critic(state_dim + action_dim, hidden_dim, action_dim)
        self.critic_target = Critic(state_dim + action_dim, hidden_dim, action_dim)
        hard_update(self.critic_target, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.critic_criterion = torch.nn.SmoothL1Loss(size_average=False) #TODO: MSELoss or SmoothL1Loss (Hubert)

    def get_exploitation_action_simulation(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).float())
        action = self.actor.forward(state).detach()
        return action.data.numpy()

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).float())
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        # TODO: implement parameter noise
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state).cpu()
        action = action.detach().numpy()[0,0]

        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.push(state, action, reward, new_state, done)

    def optimize(self):
        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss, critic_loss


# VREP INIT ************************************************************************************************************
class LabEnv:
    def __init__(self, mobRob, vrepHeadlessMode=False):
        self.clientId = -1
        self.mobRob = mobRob
        self.tolerance = 0.05
        self.chassisCollisionHandle = -1
        self.headlessMode = vrepHeadlessMode
        self.init()

    def init(self):
        if self.headlessMode:
            subprocess.Popen(['C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe', "-h", '-gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE', 'G:/GIT/MobRob/Scene/labScene.ttt'])
        else:
            subprocess.Popen(['C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe', '-gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE', 'G:/GIT/MobRob/Scene/labScene.ttt'])

        time.sleep(10)
        print('Simulation started')

        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientId = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

        if self.clientId != -1:
            print('Connected to remote API server')
        else:
            print('Failed connecting to remote API server')
            sys.exit('Could not connect')

        self.mobRob.clientId = self.clientId
        self.initSimulationObjects()

    def initSimulationObjects(self):
        self.mobRob.initMotors()
        self.mobRob.initProxSensors()
        mobRobHandleError, self.mobRob.mobRobHandle = vrep.simxGetObjectHandle(self.clientId, "MobRob",
                                                                               vrep.simx_opmode_blocking)
        mobRobCollisionHandleError, self.chassisCollisionHandle = vrep.simxGetCollisionHandle(self.clientId,
                                                                                              "MobRobChassis",
                                                                                              vrep.simx_opmode_blocking)
        self.mobRob.initReadings()
        self.getCollision(vrep.simx_opmode_streaming)

    def restart(self):
        self.stop()
        time.sleep(0.1)
        self.start()
        state = self.mobRob.getState()
        # self.pause()
        return state

    def computeReward(self, state, desiredState):
        done = False
        positionReward = 1 - norm(np.array(desiredState[:2]) - np.array(state[:2])) ** 0.4  # TODO: include yaw angle reward
        velocityReward = (1 - max(norm(np.array(state[3:5])), self.tolerance)) ** (1 / max(norm(np.array(desiredState[:2]) - np.array(state[:2])), self.tolerance))  # TODO: include yaw speed reward
        reward = positionReward * velocityReward
        if self.getCollision():
            reward -= 50
        if norm(np.array(desiredState[:2]) - np.array(state[:2])) < self.tolerance:
            reward += 10
            if norm(np.array(desiredState[3:5]) - np.array(state[3:5])) < self.tolerance:
                done = True
                reward += 10

        return reward, done

    def step(self, action, desiredState):
        self.mobRob.setMotorsTargetVelocities(action)
        # vrep.simxSynchronousTrigger(self.clientId)
        state = self.mobRob.getState()
        groundTruthState = self.mobRob.getGroundTruthState()
        vrep.simxSynchronousTrigger(self.clientId)
        reward, done = self.computeReward(groundTruthState, desiredState)

        return state, reward, done

    def getCollision(self, vrepMode=vrep.simx_opmode_buffer):
        _, collisionOccured = vrep.simxReadCollision(self.clientId, self.chassisCollisionHandle, vrepMode)
        return collisionOccured

    def start(self):
        vrep.simxSynchronous(self.clientId, 1)
        vrep.simxStartSimulation(self.clientId, vrep.simx_opmode_oneshot)  # start the simulation

    def pause(self):
        vrep.simxPauseSimulation(self.clientId, vrep.simx_opmode_oneshot)  # stop the simulation

    def stop(self):
        vrep.simxStopSimulation(self.clientId, vrep.simx_opmode_oneshot)


class MobRob:
    def __init__(self, robotName, motorsNaming, proxSensorsNaming, clientId=-1):
        self.clientId = clientId
        self.errorCodeMotors = []
        self.motors = []
        self.errorCodeProxSensors = []
        self.proxSensors = []
        self.robotName = robotName
        self.motorsNaming = motorsNaming
        self.proxSensorsNaming = proxSensorsNaming
        self.mobRobHandle = -1

    def initMotors(self):
        for motorName in self.motorsNaming:
            errorCodeMotor, motor = vrep.simxGetObjectHandle(self.clientId, motorName, vrep.simx_opmode_oneshot_wait)
            self.errorCodeMotors.append(errorCodeMotor)
            self.motors.append(motor)

    def initProxSensors(self):
        for proxSensorName in self.proxSensorsNaming:
            errorCode, proximitySensor = vrep.simxGetObjectHandle(self. clientId, proxSensorName, vrep.simx_opmode_oneshot_wait)
            self.errorCodeProxSensors.append(errorCode)
            self.proxSensors.append(proximitySensor)

    def initReadings(self):
        _, _, _ = self.getProximitySensorsReadings(vrep.simx_opmode_streaming)
        _ = self.getPosition(vrep.simx_opmode_streaming)
        _ = self.getOrientation(vrep.simx_opmode_streaming)
        _ = self.getVelocities(vrep.simx_opmode_streaming)

    def calculateDistance(self, sensorReading):
        return norm(sensorReading)

    def getState(self, vrepMode=vrep.simx_opmode_buffer):
        state = []
        _, _, proxSensorReadings = self.getProximitySensorsReadings(vrepMode)
        state = np.append(state, proxSensorReadings)
        return state  # proxySensor0...proxySensor5

    def getGroundTruthState(self, vrepMode=vrep.simx_opmode_buffer):
        state = self.getPosition(vrepMode)
        state.append(self.getOrientation(vrepMode))
        state = np.append(state, self.getVelocities(vrepMode))
        return state  # x, y, yawAngle, vx, vy, yawVel

    def getPosition(self, vrepMode):
        _, position = vrep.simxGetObjectPosition(self.clientId, self.mobRobHandle, -1, vrepMode)
        return position[:2] # returns x and y coordinates

    def getOrientation(self, vrepMode):
        _, orientation = vrep.simxGetObjectOrientation(self.clientId, self.mobRobHandle, -1, vrepMode)
        return orientation[-1] # returns yaw angle

    def getVelocities(self, vrepMode):
        _, linear, angular = vrep.simxGetObjectVelocity(self.clientId, self.mobRobHandle, vrepMode)
        velocities = linear[1:]
        velocities.append((angular[-1]))
        return velocities # returns vx, vy and yawVel

    def getProximitySensorsReadings(self, vrepMode):
        errorCodes = []
        detectionStates = []
        distanceToDetectedPoint = []

        for proxSensor in self.proxSensors:
            err_code, detectionState, detectedPoint, _, _ = vrep.simxReadProximitySensor(self.clientId, proxSensor, vrepMode)
            errorCodes.append(err_code)
            detectionStates.append(detectionStates)
            distanceToDetectedPoint.append(self.calculateDistance(detectedPoint))

        return errorCodes, detectionStates, distanceToDetectedPoint

    def setMotorsForces(self, motorsForces):
        for motorIdx in range(len(self.motors)):
            vrep.simxSetJointForce(self.clientId, self.motors[motorIdx], motorsForces[motorIdx], vrep.simx_opmode_oneshot)  # set the joint force/torque

    def setMotorsTargetVelocities(self, motorsTargetVelocities):
        for motorIdx in range(len(self.motors)):
            vrep.simxSetJointTargetVelocity(self.clientId, self.motors[motorIdx], motorsTargetVelocities[motorIdx], vrep.simx_opmode_oneshot)  # set the joint target velocity


def main():
    train = True
    vrepHeadlessMode = True
    simulate = False
    plot = False

    state_dim = 6  # TODO: simulate battery? 6 are the number of proxy sensors
    action_dim = 2
    action_space = np.array([[-2, 2], [-2, 2]])
    action_lim = [2.0, 2.0]  # 2 o/sec is the max angular speed of each motor, max. linear velocity is 0.5 m/s

    gamma = 0.99
    epsilon = .95
    tau = 0.01  # 0.001
    learning_rate = 0.001

    trials = 1000
    trial_len = 500

    desiredState = [-1.4, 0.3, -np.pi, 0.0, 0.0, 0.0]  # x, y, yawAngle, vx, vy, yawVelocity

    noise = OUNoise(action_space)

    mobRob = MobRob(['MobRob'],
                    ['leftMotor', 'rightMotor'],
                    ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                     'proximitySensor5'])
    env = LabEnv(mobRob, vrepHeadlessMode)

    if train:
        dqn_agent = DQN(env, state_dim, action_dim, action_lim, gamma=gamma, epsilon=epsilon, tau=tau, learning_rate=learning_rate)
        if dqn_agent is not None:
            print('DQN agent initialized')
        else:
            print('DQN agent failed to initialize')

        total_num_of_steps = 0
        actions = np.zeros((trials, trial_len+1, action_dim), dtype=np.float)
        loss_actor_total = []
        loss_critic_total = []
        episode_rewards = []
        for trial in range(trials):
            cur_state = env.restart()
            episode_reward = 0
            for step in range(trial_len+1):
                total_num_of_steps += 1
                # action = dqn_agent.act(cur_state, trial)
                action = dqn_agent.get_exploration_action(cur_state)
                action = noise.get_action(action, step)
                actions[trial][step] = action
                # print(action)
                new_state, reward, done = env.step(action, desiredState)

                dqn_agent.remember(cur_state, action, reward, new_state, done)

                if len(dqn_agent.memory) > dqn_agent.batch_size:
                    loss_actor, loss_critic = dqn_agent.optimize()

                cur_state = new_state
                episode_reward += reward

                if step < trial_len and done and ~env.getCollision():
                    loss_actor = loss_actor.cpu()
                    loss_critic = loss_actor.cpu()
                    print("Completed in {} trials. Episode reward: {}, actor loss: {}, critic_loss: {}".format(trial,
                                                                                                       episode_reward,
                                                                                                       loss_actor.data.numpy(),
                                                                                                       loss_critic.data.numpy()))
                    loss_actor_total.append([loss_actor.item(), total_num_of_steps])
                    loss_critic_total.append([loss_critic.item(), total_num_of_steps])
                    break

                if step == trial_len: # time budget for episode was overstepped
                    loss_actor = loss_actor.cpu()
                    loss_critic = loss_actor.cpu()
                    print("Timeout. Failed to complete in trial {}. Episode reward: {}, actor loss: {}, critic_loss: {}".format(trial,
                                                                                                                        episode_reward,
                                                                                                                        loss_actor.data.numpy(),
                                                                                                                        loss_critic.data.numpy()))
                    if len(dqn_agent.memory) >= dqn_agent.batch_size:
                        loss_actor_total.append([loss_actor.item(), total_num_of_steps])
                        loss_critic_total.append([loss_critic.item(), total_num_of_steps])
                    break

                if env.getCollision():
                    loss_actor = loss_actor.cpu()
                    loss_critic = loss_actor.cpu()
                    print("Collision. Failed to complete in trial {}. Reward: {}, actor loss: {}, critic_loss: {}".format(trial,
                                                                                                       episode_reward,
                                                                                                       loss_actor.data.numpy(),
                                                                                                       loss_critic.data.numpy()))
                    if len(dqn_agent.memory) >= dqn_agent.batch_size:
                        loss_actor_total.append([loss_actor.item(), total_num_of_steps])
                        loss_critic_total.append([loss_critic.item(), total_num_of_steps])
                    break
            episode_rewards.append([episode_reward, trial])

            gc.collect()

        torch.save(dqn_agent.actor.state_dict(), './actor.pth')
        torch.save(dqn_agent.critic.state_dict(), './critic.pth')
        np.save('actor_loss', loss_actor_total)
        np.save('critic_loss', loss_critic_total)
        np.save('episode_rewards', episode_rewards)
    elif simulate:
        dqn_agent = DQN(env, state_dim, action_dim, action_lim, gamma=gamma, epsilon=epsilon, tau=tau, learning_rate=learning_rate)
        dqn_agent.actor.load_state_dict(torch.load('./Trainings/latest/actor.pth'))
        cur_state = env.restart()

        for step in range(100000):
            # total_num_of_steps += 1
            # action = dqn_agent.act(cur_state, trial)
            action = dqn_agent.get_exploitation_action_simulation(cur_state)
            # actions[trial][step] = action
            # print(action)
            new_state, _, done = env.step(action, desiredState)
            cur_state = new_state

    elif plot:
        actor_loss = np.load('./Trainings/latest/actor_loss.npy')
        critic_loss = np.load('./Trainings/latest/critic_loss.npy')

        f, ax = plt.subplots(2, sharex=True, sharey=False)

        # # Always
        ax[0].set_ylabel("Actor loss")
        ax[1].set_ylabel("Critic loss")
        ax[-1].set_xlabel("Number of trials")

        # t = range(0, actor_loss[-1, 1])

        # Highlight the starting x axis
        ax[0].axhline(0, color="#AAAAAA")
        ax[0].plot(actor_loss[:, 1], actor_loss[:, 0])
        ax[0].grid(True)

        ax[1].axhline(0, color="#AAAAAA")
        ax[1].plot(critic_loss[:, 1], critic_loss[:, 0])
        ax[1].grid(True)

        plt.show()

if __name__ == "__main__":
    main()