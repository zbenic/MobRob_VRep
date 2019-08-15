import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size += 1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def save(self):
        np.save("replay_buffer", self.buffer)

    def load(self):
        return np.load("replay_buffer.npy", allow_pickle=True)


class EnvironmentTiles:
    def __init__(self, envWidth, envLength, envOriginCoordinate, tileSize):
        self.envWidth = envWidth
        self.envLength = envLength
        self.envOriginCoordinate = envOriginCoordinate
        self.tileSize = tileSize
        self.maxRowNum = int(np.ceil(self.envWidth / self.tileSize[1]))
        self.maxColumnNum = int(np.ceil(self.envLength / self.tileSize[0]))
        self.tiles = self.generateTiles()
        self.currentTile = ()
        self.lastTile = ()

    def generateTiles(self):
        tiles = np.empty((self.maxRowNum, self.maxColumnNum), dtype=object)

        for row in range(self.maxRowNum):
            for column in range(self.maxColumnNum):
                tileXOrigin = self.envOriginCoordinate[0] + column * self.tileSize[0]
                tileYOrigin = self.envOriginCoordinate[1] + row * self.tileSize[1]
                tiles[row, column] = Tile([tileXOrigin, tileYOrigin], self.tileSize)

        return tiles

    def getCurrentTile(self, agentPosition):
        row = int(np.ceil(self.maxRowNum    * (abs(self.envOriginCoordinate[1]) + agentPosition[1]) / self.envWidth)) - 1  # -1 is because of zero index
        column = int(np.ceil(self.maxColumnNum * (abs(self.envOriginCoordinate[0]) + agentPosition[0]) / self.envLength)) - 1  # -1 is because of zero index
        self.currentTile = (row, column)

    def updateCurrentTile(self, agentPosition):
        self.getCurrentTile(agentPosition)
        self.tiles[self.currentTile].updateTile(agentPosition)

    def updateLastVisitedTile(self, agentPosition):
        self.tiles[self.lastTile].updateTile(agentPosition)

    def updateTiles(self, agentPosition):
        self.updateCurrentTile(agentPosition)

        if not self.lastTile:
            self.lastTile = self.currentTile
        elif self.lastTile != self.currentTile:
            self.updateLastVisitedTile(agentPosition)
            self.lastTile = self.currentTile

    def tileVisited(self):
        return self.tiles[self.currentTile].visited

    def agentLeftTile(self):
        return self.tiles[self.currentTile].agentLeftTile

    def getTileEvaluationStatus(self):
        return self.tiles[self.currentTile].evaluated

    def setTileEvaluated(self):
        self.tiles[self.currentTile].evaluated = True

    def reset(self):
        for row in range(self.maxRowNum):
            for column in range(self.maxColumnNum):
                self.tiles[row, column].reset()



class Tile:
    def __init__(self, tileOrigin, tileSize):
        self.tileOrigin = tileOrigin
        self.tileXSize = tileSize[0]
        self.tileYSize = tileSize[1]
        self.agentOnTile = False
        self.agentLeftTile = False
        self.visited = False
        self.evaluated = False

    def tileVisited(self):
        if self.agentOnTile and not self.visited:
            self.visited = True

    def onTile(self, agentPosition):
        if self.tileOrigin[0] <= agentPosition[0] < (self.tileOrigin[0] + self.tileXSize) and \
           self.tileOrigin[1] <= agentPosition[1] < (self.tileOrigin[1] + self.tileYSize):
            self.agentOnTile = True
        else:
            self.agentOnTile = False

    def leftTile(self, agentPosition):
        if not (self.tileOrigin[0] <= agentPosition[0] < (self.tileOrigin[0] + self.tileXSize) and
                self.tileOrigin[1] <= agentPosition[1] < (self.tileOrigin[1] + self.tileYSize)) and \
                self.visited:
            self.agentLeftTile = True

    def updateTile(self, agentPosition):
        self.onTile(agentPosition)
        self.tileVisited()
        self.leftTile(agentPosition)

    def reset(self):
        self.agentOnTile = False
        self.agentLeftTile = False
        self.visited = False
        self.evaluated = False
