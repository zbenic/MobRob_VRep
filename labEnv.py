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
    def __init__(self, mobRob, terminalState, actionBounds, vrepHeadlessMode=False):
        self.clientId = -1
        self.mobRob = mobRob
        self.chassisCollisionHandle = -1
        self.headlessMode = vrepHeadlessMode
        self.init()
        self.collision = False
        self.terminalState = terminalState
        self.actionBounds = actionBounds

    def init(self):
        if self.headlessMode:
            subprocess.Popen(['C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe', "-h", '-gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE', 'C:/visageGIT/MobRob_VRep/Scene/labSceneComplex.ttt'])
        else:
            subprocess.Popen(['C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe', '-gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE', 'C:/visageGIT/MobRob_VRep/Scene/labSceneComplex.ttt'])

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
        self.mobRob.initDone()
        self.getCollision(vrep.simx_opmode_streaming)

    def reset(self):
        self.stop()
        time.sleep(0.1)
        self.start()
        self.collision = False
        passed, state = self.mobRob.getState()
        # self.pause()
        return passed, state

    def computeReward(self, state, action):
        done = False

        forwardRew = state[3]

        lb, ub = self.actionBounds
        scaling = (ub - lb) * 0.5
        controlCost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        surviveReward = 0.05

        reward = forwardRew - controlCost + surviveReward

        if self.terminalStateAchieved(state):
            reward += 100
            done = True

        if self.collision:
            reward -= 50

        return reward, done

    def terminalStateAchieved(self, state):
        if self.terminalState[0] <= state[0] <= self.terminalState[1] and \
           self.terminalState[2] <= state[1] <= self.terminalState[3]:
            return True
        else:
            return False

    def step(self, action):
        self.mobRob.setMotorsTargetVelocities(action)
        vrep.simxSynchronousTrigger(self.clientId)
        passed, state = self.mobRob.getState()
        if passed:
            groundTruthState = self.mobRob.getGroundTruthState(state)
            self.getCollision()
            reward, done = self.computeReward(groundTruthState, action)
        else:
            reward = 0
            done = False

        return state, reward, done, passed

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
        self.minProxSensorDistance = 0.1
        self.maxProxSensorDistance = 0.8
        self.mobRobHandle = -1
        self.transformationMatrix = []
        self.invertedTransformationMatrix = []
        self.initializationDone = False

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

    def initDone(self):
        self.initializationDone = True

    def calculateDistance(self, sensorReading):
        return norm(sensorReading)

    def getState(self, vrepMode=vrep.simx_opmode_buffer):
        passed, self.invertedTransformationMatrix = self.getInvertedTransformationMatrix()
        state = []
        if passed == False:
            return False, state
        _, _, proxSensorReadings = self.getProximitySensorsReadings(vrepMode)
        proxSensorReadings = np.clip(proxSensorReadings, self.minProxSensorDistance, self.maxProxSensorDistance)
        position = self.getPosition(vrepMode)
        orientation = self.getOrientation(vrepMode)
        velocities = self.getVelocities(vrepMode)
        state = np.append(state, position)  # TODO: just for test run
        state = np.append(state, orientation)
        state = np.append(state, velocities)   # TODO: just for test run
        # state = np.append(state, state - desiredState)  # error vector
        state = np.append(state, proxSensorReadings)
        return True, state  # x,y,yaw,v_x,v_y,v_yaw,e_x,e_y,e_yaw,e_vx,e_vy,e_vyaw,proxySensor0...proxySensor5

    def getGroundTruthState(self, state):
        groundTruthState = state[:2]  # x and y positions
        groundTruthState = np.append(groundTruthState, state[2])  # yaw angle
        groundTruthState = np.append(groundTruthState, state[3:6])  # vx and vy
        return groundTruthState  # x, y, yawAngle, vx, vy, yawVel

    def getPosition(self, vrepMode):
        _, position = vrep.simxGetObjectPosition(self.clientId, self.mobRobHandle, -1, vrepMode)
        return position[:2] # returns x and y coordinates

    def getOrientation(self, vrepMode):
        _, orientation = vrep.simxGetObjectOrientation(self.clientId, self.mobRobHandle, -1, vrepMode)
        return orientation[-1] # returns yaw angle

    def getVelocities(self, vrepMode):
        _, linear, angular = vrep.simxGetObjectVelocity(self.clientId, self.mobRobHandle, vrepMode)
        if self.initializationDone:
            linear = self.getLinearVelocityWrtRobotFrame(linear)
            linear = list(np.array(linear).reshape(-1,))
        velocities = linear[:2]
        velocities.append((angular[-1]))
        return velocities  # returns vx, vy (in robot frame) and yawVel (in world frame)

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

    def getTransformationMatrix(self):
        [_, _, retFloats, _, _] = vrep.simxCallScriptFunction(self.clientId, 'HelperScripts',
                                                              vrep.sim_scripttype_childscript,
                                                              'getMobRobMatrix', [], [], [],
                                                              '',
                                                              vrep.simx_opmode_blocking)
        transformationMat = np.matrix(retFloats).reshape((3, 4))
        lastRow = np.matrix([0, 0, 0, 1])
        transformationMat = np.vstack([transformationMat, lastRow])
        return transformationMat

    def getInvertedTransformationMatrix(self):
        passed = True
        [_, _, retFloats, _, _] = vrep.simxCallScriptFunction(self.clientId, 'HelperScripts',
                                                              vrep.sim_scripttype_childscript,
                                                              'getInvertedMobRobMatrix', [], [], [],
                                                              '',
                                                              vrep.simx_opmode_blocking)

        invertedTransformationMat = np.zeros(1)

        try:
            invertedTransformationMat = np.matrix(retFloats).reshape((3, 4))
            invertedTransformationMat = np.transpose(invertedTransformationMat)
            invertedTransformationMat = np.delete(invertedTransformationMat, 3, 0)
        except ValueError:
            passed = False
            invertedTransformationMat = np.zeros(1)

        return passed, invertedTransformationMat

    def getLinearVelocityWrtRobotFrame(self, linearVelocities):
        return np.dot(self.invertedTransformationMatrix, linearVelocities)

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
