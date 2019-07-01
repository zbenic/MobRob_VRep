import numpy as np
from numpy.linalg import norm
import subprocess
import sys
import time
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

class LabEnv:
    def __init__(self, mobRob, vrepHeadlessMode=False):
        self.clientId = -1
        self.mobRob = mobRob
        self.tolerance = 0.05
        self.chassisCollisionHandle = -1
        self.headlessMode = vrepHeadlessMode
        self.init()
        self.collision = False

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
        time.sleep(2)

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

    def reset(self, desiredState):
        self.stop()
        time.sleep(0.1)
        self.start()
        self.collision = False
        state = self.mobRob.getState(desiredState)
        # self.pause()
        return state

    def computeReward(self, state, desiredState):
        done = False
        a = 1
        lamb = 0.75
        # Lamb = np.diag([1, 0.75, 0.75, 0.25, 1])
        # positionReward = 1 - norm(np.array(desiredState[:2]) - np.array(state[:2])) ** 0.4  # TODO: include yaw angle reward
        # velocityReward = (1 - max(norm(np.array(state[3:5])), self.tolerance)) ** (1 / max(norm(np.array(desiredState[:2]) - np.array(state[:2])), self.tolerance))  # TODO: include yaw speed reward
        # reward = positionReward * 10
        reward = lamb * np.exp(-1 / a**2 * np.transpose(state[:7] - desiredState).dot(state[:7] - desiredState))
        if self.collision:
            reward -= 50
        if norm(np.array(desiredState[:2]) - np.array(state[:2])) < self.tolerance:
            reward += 30
            print("Position reached!")
            done = True
            if norm(np.array(desiredState[3:5]) - np.array(state[3:5])) < self.tolerance:
                print("Speed reached!")
                done = True
                reward += 50

        return reward, done

    def step(self, action, desiredState):
        self.mobRob.setMotorsTargetVelocities(action)
        vrep.simxSynchronousTrigger(self.clientId)
        state = self.mobRob.getState(desiredState)
        groundTruthState = self.mobRob.getGroundTruthState()
        # vrep.simxSynchronousTrigger(self.clientId)
        self.getCollision()
        reward, done = self.computeReward(groundTruthState, desiredState)

        return state, reward, done

    def getCollision(self, vrepMode=vrep.simx_opmode_buffer):
        _, collisionOccured = vrep.simxReadCollision(self.clientId, self.chassisCollisionHandle, vrepMode)
        self.collision = collisionOccured
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

    def getState(self, desiredState, vrepMode=vrep.simx_opmode_buffer):
        state = []
        _, _, proxSensorReadings = self.getProximitySensorsReadings(vrepMode)
        position = self.getPosition(vrepMode)
        orientation = self.getOrientation(vrepMode)
        velocities = self.getVelocities(vrepMode)
        # state = np.append(state, position)  # TODO: just for test run
        # state = np.append(state, orientation)
        # state = np.append(state, velocities)   # TODO: just for test run
        # state = np.append(state, state - desiredState)  # error vector
        state = np.append(state, proxSensorReadings)
        return state  # x,y,yaw,v_x,v_y,v_yaw,e_x,e_y,e_yaw,e_vx,e_vy,e_vyaw,proxySensor0...proxySensor5

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
        velocities = linear[:2]
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
        '''

        :param motorsTargetVelocities: unit is sec^(-1)
        :return: N/A
        '''

        for motorIdx in range(len(self.motors)):
            vrep.simxSetJointTargetVelocity(self.clientId, self.motors[motorIdx], np.deg2rad(motorsTargetVelocities[motorIdx]*360), vrep.simx_opmode_oneshot)  # set the joint target velocity
