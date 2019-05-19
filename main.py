# Make sure to have the server side running in V-REP:
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

# edit usrset.txt to turn on debugging
# debug log file debugLog.txt

import sys
import subprocess
import time
import array
from PIL import Image
from numpy.linalg import norm
import numpy as np

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


def initMotors(clientId, numberOfMotors:int = 2):
    errorCodeMotors = []
    motors = []
    motorNaming = {0: 'leftMotor',
                   1: 'rightMotor'}

    for motorIdx in range(numberOfMotors):
        errorCodeMotor, motor = vrep.simxGetObjectHandle(clientId, motorNaming[motorIdx], vrep.simx_opmode_oneshot_wait)
        errorCodeMotors.append(errorCodeMotor)
        motors.append(motor)

    return errorCodeMotors, motors


def initProxSensors(clientId, numberOfSensors:int = 6):
    errorCodeProxSensors = []
    proxSensors = []

    for proxSensorIdx in range(numberOfSensors):
        proxSensorName = "proximitySensor" + str(proxSensorIdx)
        errorCode, proximitySensor = vrep.simxGetObjectHandle(clientId, proxSensorName, vrep.simx_opmode_oneshot_wait)
        errorCodeProxSensors.append(errorCode)
        proxSensors.append(proximitySensor)

    return errorCodeProxSensors, proxSensors


def calculateDistance(sensorReading):
    return norm(sensorReading)


def getProximitySensorsReadings(clientId, proxSensors, vrepMode = vrep.simx_opmode_buffer):
    errorCodes = []
    detectionStates = []
    detectedPoints = []

    for proxSensor in proxSensors:
        err_code, detectionState, detectedPoint, _, _ = vrep.simxReadProximitySensor(clientId, proxSensor, vrepMode)
        errorCodes.append(err_code)
        detectionStates.append(detectionStates)
        detectedPoints.append(detectedPoint)

    return errorCodes, detectionStates, detectedPoints


def setMotorsForces(clientId, motors, leftMotorForce, rightMotorForce):
    vrep.simxSetJointForce(clientId, motors[0], leftMotorForce, vrep.simx_opmode_oneshot)  # set the left joint force/torque
    vrep.simxSetJointForce(clientId, motors[1], rightMotorForce, vrep.simx_opmode_oneshot)  # set the right joint force/torque


def setMotorsTargetVelocities(clientId, motors, leftMotorTargetVelocity, rightMotorTargetVelocity):
    vrep.simxSetJointTargetVelocity(clientId, motors[0], leftMotorTargetVelocity, vrep.simx_opmode_oneshot) # set the joint target velocity
    vrep.simxSetJointTargetVelocity(clientId, motors[1], rightMotorTargetVelocity, vrep.simx_opmode_oneshot) # set the joint target velocity


subprocess.Popen(['C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe', '-gREMOTEAPISERVERSERVICE_19997_FALSE_TRUE', 'G:/GIT/MobRob/Scene/labScene.ttt'])
time.sleep(10)

print('Program started')
vrep.simxFinish(-1)  # just in case, close all opened connections
clientId = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

if clientId != -1:
    print('Connected to remote API server')
else:
    print('Failed connecting to remote API server')
    sys.exit('Could not connect')

startTime = time.time()

errorCodeMotors, motors = initMotors(clientId)
# errorCodeCamera, frontRobotCamera = vrep.simxGetObjectHandle(clientId, 'frontRobotCamera', vrep.simx_opmode_oneshot_wait)
errorCodeInitProxSensors, proxSensors = initProxSensors(clientId)

vrep.simxSynchronous(clientId, 1) # enable the synchronous mode (client side). The server side (i.e. V-REP) also needs to be enabled.
vrep.simxStartSimulation(clientId, vrep.simx_opmode_oneshot) # start the simulation

# TODO: set joint force/torque from EMIR documentation
setMotorsForces(clientId, motors, 100, 100)

simStep = 0
# errorCodeCameraImage, frontCameraResolution, frontCameraImage = vrep.simxGetVisionSensorImage(clientId, frontRobotCamera, 0, vrep.simx_opmode_streaming)

errorCodeProxSensors, detectionStates, detectedPoints = getProximitySensorsReadings(clientId, proxSensors, vrep.simx_opmode_streaming)

while simStep < 10000:
    vrep.simxSetJointTargetVelocity(clientId, motors[0], 5, vrep.simx_opmode_oneshot) # set the joint target velocity
    vrep.simxSetJointTargetVelocity(clientId, motors[1], 5, vrep.simx_opmode_oneshot) # set the joint target velocity

    errorCodeProx, detectionStatesProx, detectedPointsProx = getProximitySensorsReadings(clientId, proxSensors)

    vrep.simxSynchronousTrigger(clientId) # trigger next simulation step. Above commands will be applied

    simStep += 1

vrep.simxPauseSimulation(clientId, vrep.simx_opmode_oneshot) # stop the simulation


# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
# You can guarantee this with (for example):
vrep.simxGetPingTime(clientId)

# Now close the connection to V-REP:
vrep.simxFinish(clientId)


print ('Program ended')



# errorCodeCameraImage, frontCameraResolution, frontCameraImage = vrep.simxGetVisionSensorImage(clientId, frontRobotCamera, 0, vrep.simx_opmode_buffer)
# if errorCodeCameraImage == vrep.simx_return_ok:
#     img = np.array(frontCameraImage, dtype=np.uint8)
#     img = img.reshape([frontCameraResolution[1], frontCameraResolution[0], 3])
#     img = Image.fromarray(img, 'RGB')
# img.save('my.png')
# img.show()

# this should return image to camera. See https://github.com/nemilya/vrep-api-python-opencv/blob/master/handle_vision_sensor.py
# if errorCodeCameraImage == vrep.simx_return_ok:
#     errorCodeCameraImage = vrep.simxSetVisionSensorImage(clientId, v1, image, 0, vrep.simx_opmode_oneshot)