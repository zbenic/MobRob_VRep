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

import sys
import subprocess
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

subprocess.call("C:/Program Files/V-REP3/V-REP_PRO_EDU/vrep.exe -gREMOTEAPISERVERSERVICE_19997_TRUE_TRUE G:/GIT/MobRob/Scene/labScene.ttt")
time.sleep(5)

print('Program started')
vrep.simxFinish(-1)  # just in case, close all opened connections
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP

if clientID != -1:
    print('Connected to remote API server')
else:
    print('Failed connecting to remote API server')
    sys.exit('Could not connect')

vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

startTime = time.time()

errorCode, leftMotor = vrep.simxGetObjectHandle(clientID, "leftMotor", vrep.simx_opmode_oneshot_wait)
errorCode, rightMotor = vrep.simxGetObjectHandle(clientID, "rightMotor", vrep.simx_opmode_oneshot_wait)

vrep.simxSetJointTargetVelocity(clientID, leftMotor, 0.2, vrep.simx_opmode_oneshot)

time.sleep(2)


# Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
# You can guarantee this with (for example):
vrep.simxGetPingTime(clientID)

# Now close the connection to V-REP:
vrep.simxFinish(clientID)


print ('Program ended')