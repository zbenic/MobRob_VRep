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

errorCodeMotorLeft, leftMotor = vrep.simxGetObjectHandle(clientId, "leftMotor", vrep.simx_opmode_oneshot_wait)
errorCodeMotorRight, rightMotor = vrep.simxGetObjectHandle(clientId, "rightMotor", vrep.simx_opmode_oneshot_wait)
errorCodeCamera, frontRobotCamera = vrep.simxGetObjectHandle(clientId, 'frontRobotCamera', vrep.simx_opmode_oneshot_wait)

vrep.simxSynchronous(clientId, 1) # enable the synchronous mode (client side). The server side (i.e. V-REP) also needs to be enabled.
vrep.simxStartSimulation(clientId, vrep.simx_opmode_oneshot) # start the simulation
vrep.simxSetJointForce(clientId, leftMotor, 1.0, vrep.simx_opmode_oneshot) # set the joint force/torque
vrep.simxSetJointForce(clientId, rightMotor, 1.0, vrep.simx_opmode_oneshot) # set the joint force/torque

simStep = 0
errorCodeCameraImage, frontCameraResolution, frontCameraImage = vrep.simxGetVisionSensorImage(clientId, frontRobotCamera, 0, vrep.simx_opmode_streaming)

while(simStep < 100):
    vrep.simxSetJointTargetVelocity(clientId, leftMotor, 5, vrep.simx_opmode_oneshot) # set the joint target velocity
    vrep.simxSetJointTargetVelocity(clientId, rightMotor, 5, vrep.simx_opmode_oneshot) # set the joint target velocity
    vrep.simxSynchronousTrigger(clientId) # trigger next simulation step. Above commands will be applied

    errorCodeCameraImage, frontCameraResolution, frontCameraImage = vrep.simxGetVisionSensorImage(clientId, frontRobotCamera, 0, vrep.simx_opmode_buffer)
    if errorCodeCameraImage == vrep.simx_return_ok:
        img = np.array(frontCameraImage, dtype=np.uint8)
        img = img.reshape([frontCameraResolution[1], frontCameraResolution[0], 3])
        img = Image.fromarray(img, 'RGB')
        img.save('my.png')
        img.show()

    # this should return image to camera. See https://github.com/nemilya/vrep-api-python-opencv/blob/master/handle_vision_sensor.py
    # if errorCodeCameraImage == vrep.simx_return_ok:
    #     errorCodeCameraImage = vrep.simxSetVisionSensorImage(clientId, v1, image, 0, vrep.simx_opmode_oneshot)

    simStep += 1

vrep.simxPauseSimulation(clientId, vrep.simx_opmode_oneshot) # stop the simulation


# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
# You can guarantee this with (for example):
vrep.simxGetPingTime(clientId)

# Now close the connection to V-REP:
vrep.simxFinish(clientId)


print ('Program ended')