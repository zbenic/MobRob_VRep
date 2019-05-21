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

train = True
simulate = False
plot = False

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Torch will use " + device.__str__() + " as device.")

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        # self.dt = dt
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

    # def sample(self):
    #     dx = self.theta * (self.mu - self.X) * self.dt
    #     dx = dx + self.sigma * np.sqrt(self.dt) * np.random.normal(len(self.X))
    #     self.X = self.X + dx
    #     return self.X


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


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)



class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 256).to(device)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size()).cuda()
        self.fcs2 = nn.Linear(256, 128).to(device)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size()).cuda()

        self.fca1 = nn.Linear(action_dim, 128).to(device)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size()).cuda()

        self.fc2 = nn.Linear(256, 128).to(device)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size()).cuda()

        self.fc3 = nn.Linear(128, 1).to(device)
        self.fc3.weight.data.uniform_(-EPS, EPS).cuda()

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        state = state.cuda()
        action = action.cuda()
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 256).to(device)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size()).cuda()

        self.fc2 = nn.Linear(256, 128).to(device)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size()).cuda()

        self.fc3 = nn.Linear(128, 64).to(device)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size()).cuda()

        self.fc4 = nn.Linear(64, action_dim).to(device)
        self.fc4.weight.data.uniform_(-EPS, EPS).cuda()

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        state = state.cuda()
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))

        return action * torch.from_numpy(np.array(self.action_lim))


class DQN:
    def __init__(self, env, state_dim, action_dim, action_lim, gamma=0.85, epsilon=1.0, tau=0.125, learning_rate=0.005):
        self.env = env
        self.min_pool_size = 10000
        self.batch_size = 128
        self.memory = deque(maxlen=1000000)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.tau = tau

        self.input_num = state_dim
        self.output_num = action_dim
        self.action_lim = action_lim

        self.noise = OrnsteinUhlenbeckActionNoise(self.output_num)

        self.actor = Actor(state_dim, action_dim, action_lim)
        self.target_actor = Actor(state_dim, action_dim, action_lim)
        hard_update(self.target_actor, self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate/10)

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)


        self.criterion = torch.nn.SmoothL1Loss(size_average=False) #TODO: MSELoss or SmoothL1Loss (Hubert)

    # def create_actor(self):
    #     model = torch.nn.Sequential(
    #         torch.nn.Linear(self.input_num, 64),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(64, 64),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(64, self.output_num),
    #     )
    #     return model

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
        state = Variable(torch.from_numpy(state).float())
        action = self.actor.forward(state).detach().cpu()
        new_action = action.data.numpy() + self.noise.sample()
        return new_action

    def act(self, state, trial):
        if trial % 5 == 0:
            # validate every 5th episode
            action = self.get_exploitation_action(state)
        else:
            # get action based on observation, use exploration policy here
            action = self.get_exploration_action(state)
        return action #.clip(-1, 1)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def unpack_batch(self, batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []
        for exp in batch:
            state = exp[0]
            states.append(state)
            actions.append(exp[1])
            rewards.append(exp[2])
            dones.append(exp[4])
            if exp[3] is None:
                last_states.append(state)  # the result will be masked anyway
            else:
                last_states.append(exp[3])
        return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)

    def calc_loss(self, batch):
        states, actions, rewards, dones, next_states = self.unpack_batch(batch)

        states_v = torch.tensor(states, dtype=torch.float32)
        next_states_v = torch.tensor(next_states, dtype=torch.float32)
        actions_v = torch.tensor(actions, dtype=torch.float32)
        rewards_v = torch.tensor(rewards, dtype=torch.float32)
        done_mask = torch.ByteTensor(dones)

        # if cuda:
        #     states_v = states_v.cuda(non_blocking=cuda_async)
        #     next_states_v = next_states_v.cuda(non_blocking=cuda_async)
        #     actions_v = actions_v.cuda(non_blocking=cuda_async)
        #     rewards_v = rewards_v.cuda(non_blocking=cuda_async)
        #     done_mask = done_mask.cuda(non_blocking=cuda_async)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        next_actions_v = self.target_actor.forward(next_states_v).detach()
        next_val = torch.squeeze(self.target_critic.forward(next_states_v, next_actions_v).detach())

        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = rewards_v + self.gamma * done_mask.float() * next_val

        self.critic_optimizer.zero_grad()

        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(states_v, actions_v))

        # compute critic loss
        loss_critic = self.criterion(y_predicted, y_expected)

        # compute actor loss
        self.actor_optimizer.zero_grad()
        pred_a1 = self.actor.forward(states_v)
        # loss_actor = -1 * torch.sum(self.critic.forward(states_v, pred_a1))
        loss_actor = -1 * self.critic.forward(states_v, pred_a1)
        loss_actor = loss_actor.mean()

        return loss_actor, loss_critic

    def train(self, step, cuda=False):
        if len(self.memory) < self.min_pool_size:
            return torch.Tensor((1,1), None), torch.Tensor((1,1), None)

        batch = random.sample(self.memory, self.batch_size)

        loss_actor, loss_critic = self.calc_loss(batch)

        # if step % 100 == 0:
        #     print('Iteration :- ', step, ' Loss_actor :- ', loss_actor.data.numpy(),
        #           ' Loss_critic :- ', loss_critic.data.numpy())

        # -------------------- update the critic --------------------

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        # self.critic_optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_critic.backward()
        # Clamp the gradients to avoid vanishing gradient problem
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        # Calling the step function on an Optimizer makes an update to its parameters
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        # self.actor_optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_actor.backward()
        # Clamp the gradients to avoid the vanishing gradient problem
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        # Calling the step function on an Optimizer makes an update to its parameters
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, self.tau)  # updates actor target model
        soft_update(self.target_critic, self.critic, self.tau)  # updates actor target model

        return loss_actor, loss_critic

    # def save_model(self, fn):
    #     self.actor.save(fn)

    def optimize(self):
        if len(self.memory) < self.min_pool_size:
            return torch.Tensor(), torch.Tensor()

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, dones, next_states = self.unpack_batch(batch)

        states_v = torch.tensor(states, dtype=torch.float32)
        next_states_v = torch.tensor(next_states, dtype=torch.float32)
        actions_v = torch.tensor(actions, dtype=torch.float32)
        rewards_v = torch.tensor(rewards, dtype=torch.float32)
        done_mask = torch.ByteTensor(dones)

        if torch.cuda.is_available():
            states_v = states_v.cuda()
            next_states_v = next_states_v.cuda()
            actions_v = actions_v.cuda()
            rewards_v = rewards_v.cuda()
            done_mask = done_mask.cuda()

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        next_actions_v = self.target_actor.forward(next_states_v).detach()
        next_val = torch.squeeze(self.target_critic.forward(next_states_v, next_actions_v).detach())

        # y_exp = r + gamma*Q'( s2, pi'(s2))
        y_expected = rewards_v + (1-done_mask.float()) * self.gamma * next_val #TODO: * done_mask.float() or (1-isdone) needed?

        # y_pred = Q( s1, a1)
        y_predicted = torch.squeeze(self.critic.forward(states_v, actions_v))

        # compute critic loss
        loss_critic = self.criterion(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        pred_a1 = self.actor.forward(states_v)
        # pred_q1 = torch.squeeze(self.critic.forward(states_v, pred_a1)) #TODO: try this with (https://github.com/ctmakro/gymnastics/blob/master/ddpg.py)
        # loss_actor = -1 * torch.sum(self.critic.forward(states_v, pred_a1))
        loss_actor = -1 * self.critic.forward(states_v, pred_a1)
        loss_actor = loss_actor.mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, self.tau)  # updates actor target model
        soft_update(self.target_critic, self.critic, self.tau)  # updates actor target model

        return loss_actor, loss_critic


# VREP INIT ************************************************************************************************************
class LabEnv:
    def __init__(self, MobRob):
        self.clientId = -1
        self.mobRob = MobRob
        self.tolerance = 0.1
        self.chassisCollisionHandle = -1

    def init(self):
        subprocess.Popen(['C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe', '-gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE',
                          'G:/GIT/MobRob/Scene/labScene.ttt'])
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
        self.mobRob.mobRobHandle = vrep.simxGetObjectHandle(self.clientId, "MobRob", vrep.simx_opmode_blocking)
        self.chassisCollisionHandle = vrep.simxGetCollisionHandle(self.clientId, "MobRobChassis",
                                                                  vrep.simx_opmode_blocking)

    def initSimulationObjects(self):
        self.mobRob.initMotors()
        self.mobRob.initProxSensors()


    def restart(self):
        state = self.mobRob.getState()
        self.stop()
        self.start()
        return state

    def computeReward(self, state, desiredState):
        done = False
        alpha = 1
        beta = 1
        positionReward = (norm(desiredState[:3]) - norm(state[:3]))
        velocityReward = (norm(desiredState[3:6]) - norm(state[3:6]))
        reward = np.exp(-alpha * positionReward) + np.exp(-beta * velocityReward)
        if (norm(desiredState[:3]) - norm(state[:3])) < self.tolerance:
            reward += 10
            if (norm(desiredState[3:6]) - norm(state[3:6])) < self.tolerance:
                done = True
                reward += 10

        return reward, done

    def step(self, action, desiredState):
        self.mobRob.setMotorsTargetVelocities(action)
        vrep.simxSynchronousTrigger(self.clientId)
        _, _, state = self.mobRob.getProximitySensorsReadings()
        reward, done = self.computeReward(state, desiredState)

        return state, reward, done

    def getCollision(self, vrepMode=vrep.simx_opmode_buffer):
        _, collisionOccured = vrep.simxReadCollision(self.clientId, "MobRobChassis", vrepMode)
        return collisionOccured

    def start(self):
        vrep.simxStartSimulation(self.clientId, vrep.simx_opmode_oneshot)  # start the simulation

    def pause(self):
        vrep.simxPauseSimulation(self.clientId, vrep.simx_opmode_oneshot)  # stop the simulation

    def stop(self):
        vrep.simxStopSimulation(self.clientId, vrep.simx_opmode_oneshot)


class MobRob:
    def __init__(self, robotName, motorsNaming, proxSensorsNaming, clientId = -1):
        self.clientId = clientId
        self.errorCodeMotors = []
        self.motors = []
        self.errorCodeProxSensors = []
        self.proxSensors = []
        self.robotName = robotName
        self.motorsNaming = motorsNaming
        self.proxSensorsNaming = proxSensorsNaming
        self.mobRobHandle = -1
        self.initMotors()
        self.initProxSensors()

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

    def calculateDistance(self, sensorReading):
        return norm(sensorReading)

    def getState(self, vrepMode=vrep.simx_opmode_buffer):
        state = self.getPosition(vrepMode)
        state.append(self.getOrientation(vrepMode))
        state.append((self.getVelocities(vrepMode)))
        state.append(self.getProximitySensorsReadings(vrepMode))
        return state  # x, y, yawAngle, vx, vy, yawVel, proxySensor0...proxySensor5

    def getPosition(self, vrepMode):
        _, position = vrep.simxGetObjectPosition(self.clientId, self.mobRobHandle, -1, vrepMode)
        return position[1:] # returns x and y coordinates

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
    gamma = 0.99
    epsilon = .95
    tau = 0.001  # 0.125
    learning_rate = 0.001

    trials = 20000
    trial_len = 290

    desiredState = [1.4, 0.3, -180.0, 0.0, 0.0, 0.0]  # x, y, yawAngle, vx, vy, yawVelocity

    mobRob = MobRob(['MobRob'],
                    ['leftMotor', 'rightMotor'],
                    ['proximitySensor0', 'proximitySensor1', 'proximitySensor2', 'proximitySensor3', 'proximitySensor4',
                     'proximitySensor5'])

    env = LabEnv(mobRob)

    state_dim = 6  # TODO: simulate battery? 6 are the number of proxy sensors
    action_dim = 2
    action_lim = [2, 2]  # 2 o/sec is the max angular speed of each motor

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
        for trial in range(trials):
            cur_state = env.restart()
            for step in range(trial_len+1):
                total_num_of_steps += 1
                # action = dqn_agent.act(cur_state, trial)
                action = dqn_agent.get_exploration_action(cur_state)
                actions[trial][step] = action
                # print(action)
                new_state, reward, done = env.step(action, desiredState)

                if step < trial_len and done and ~env.getCollision():
                    loss_actor = loss_actor.cpu()
                    loss_critic = loss_actor.cpu()
                    print("Completed in {} trials. Reward: {}, actor loss: {}, critic_loss: {}".format(trial,
                                                                                                       reward,
                                                                                                       loss_actor.data.numpy(),
                                                                                                       loss_critic.data.numpy()))
                    break
                    # dqn_agent.save_model("success.model")
                    # sys.exit()

                dqn_agent.remember(cur_state, action, reward, new_state, done)

                # loss_actor, loss_critic = dqn_agent.train(step)  # internally iterates default (prediction) model
                loss_actor, loss_critic = dqn_agent.optimize()  # internally iterates default (prediction) model


                # if step % update_target_network_step == 0:
                #     soft_update(dqn_agent.target_actor, dqn_agent.actor, tau)  # updates actor target model
                #     soft_update(dqn_agent.target_critic, dqn_agent.critic, tau)  # updates actor target model

                cur_state = new_state
                # if done:
                #     break

                # if step >= 199 and step % 10 == 0:
                #     dqn_agent.save_model("trial-{}.model".format(trial))

                if step == trial_len: # time budget for episode was overstepped
                    loss_actor = loss_actor.cpu()
                    loss_critic = loss_actor.cpu()
                    print("Timeout. Failed to complete in trial {}. Reward: {}, actor loss: {}, critic_loss: {}".format(trial,
                                                                                                       reward,
                                                                                                       loss_actor.data.numpy(),
                                                                                                       loss_critic.data.numpy()))
                    if len(dqn_agent.memory) >= dqn_agent.min_pool_size:
                        loss_actor_total.append([loss_actor.item(), total_num_of_steps])
                        loss_critic_total.append([loss_critic.item(), total_num_of_steps])
                    break

                if env.getCollision():
                    loss_actor = loss_actor.cpu()
                    loss_critic = loss_actor.cpu()
                    print("Collision. Failed to complete in trial {}. Reward: {}, actor loss: {}, critic_loss: {}".format(trial,
                                                                                                       reward,
                                                                                                       loss_actor.data.numpy(),
                                                                                                       loss_critic.data.numpy()))
                    if len(dqn_agent.memory) >= dqn_agent.min_pool_size:
                        loss_actor_total.append([loss_actor.item(), total_num_of_steps])
                        loss_critic_total.append([loss_critic.item(), total_num_of_steps])
                    break

            gc.collect()

        torch.save(dqn_agent.actor.state_dict(), './actor.pth')
        torch.save(dqn_agent.critic.state_dict(), './critic.pth')
        np.save('actor_loss', loss_actor_total)
        np.save('critic_loss', loss_critic_total)
    elif simulate:
        dqn_agent = DQN(env, state_dim, action_dim, action_lim, gamma=gamma, epsilon=epsilon, tau=tau, learning_rate=learning_rate)
        dqn_agent.actor.load_state_dict(torch.load('actor.pth'))
        cur_state = env.reset()
        env.render()
        time.sleep(5)

        for step in range(100000):
            # total_num_of_steps += 1
            # action = dqn_agent.act(cur_state, trial)
            action = dqn_agent.get_exploitation_action_simulation(cur_state)
            # actions[trial][step] = action
            # print(action)
            new_state, _, done, _ = env.step(action)
            cur_state = new_state

    elif plot:
        actor_loss = np.load('actor_loss.npy')
        critic_loss = np.load('critic_loss.npy')

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