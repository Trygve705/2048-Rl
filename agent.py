import torch
import torch.nn as nn
import torch.optim as optim


import random
from collections import deque
import numpy as np


class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        """
        # For nytt nettverk

        self.network = nn.Sequential(
           nn.Linear(16, 128),
           nn.ReLU(),
           nn.Linear(128, 256),
           nn.ReLU(),
           nn.Linear(256, 512),
           nn.ReLU(),
           nn.Linear(512, 1024),
           nn.ReLU(),
           nn.Linear(1024, 512),
           nn.ReLU(),
           nn.Linear(512, 256),
           nn.ReLU(),
           nn.Linear(256, 128),
           nn.ReLU(),
           nn.Linear(128, 4)
        )"""


        # For best model forbedret
        self.network = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        


    def forward(self, state):
        state = state.flatten(start_dim = 1)
        return self.network(state)


class agent:


    def __init__(self):
        self.gamma = 0.99

        self.epsilon = 1.00
        self.epsilonMinimum = 0.05
        self.epsilonDecay = 0.9999

        self.batchSize = 128
        self.memory = deque(maxlen=100_000)

        self.model = DQN()
        self.targetModel = DQN()

        self.targetModel.load_state_dict(
            self.model.state_dict()
        )

        self.targetModel.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0002
        )

        self.lossFunction = nn.SmoothL1Loss()

    def chooseAction(self, state, validMoves):
        processedState = self.preprocessState(state)
        validAction = []

        for action in range(4):
            if validMoves[action] == True:
                validAction.append(action)

        # Velge tilfeldig
        randomNumber = random.random()

        if randomNumber < self.epsilon:
            chosenAction = random.choice(validAction)
            return chosenAction

        # Omgjøre brettet
        stateTensor = torch.tensor(
            processedState, 
            dtype = torch.float32
        )   

        stateTensor = stateTensor.unsqueeze(0)

        with torch.no_grad():
            qValues = self.model(stateTensor)[0]

        for action in range(4):
            if validMoves[action] == False:
                qValues[action] = float("-inf")

        chosenAction = torch.argmax(qValues).item()

        return chosenAction

    def remember(self, state, action, reward, nextState, done, nextValidMoves):
        nextValidMoves = list(nextValidMoves)
        experience = (
            state.copy(),
            action,
            reward,
            nextState.copy(),
            done,
            nextValidMoves.copy()
        )

        self.memory.append(experience)


    def learn (self):
        if len(self.memory) < 1000:
            return

        batch = random.sample(self.memory, self.batchSize)

        states = []
        actions = []
        rewards = []
        nextStates = []
        dones = []
        nextValidMovesList = []

        for experience in batch:
            state, action, reward, nextState, done, nextValidMoves = experience

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            nextStates.append(nextState)
            dones.append(done)
            nextValidMovesList.append(nextValidMoves)

        # Gjør listene om til tensorer
        states = torch.tensor(
            np.array([
            self.preprocessState(experience[0])
            for experience in batch
            ]))

        actions = torch.tensor(
            actions,
            dtype=torch.long
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        )

        nextStates = torch.tensor(
            np.array([
            self.preprocessState(experience[3])
            for experience in batch
        ]))

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        )

        nextValidMoves = torch.tensor(
            np.array(nextValidMovesList),
            dtype=torch.bool
        )

        allQValues = self.model(states)

        currentQValues = allQValues.gather(
            1, 
            actions.unsqueeze(1)
        ).squeeze(1)


        with torch.no_grad():

            nextQValues = self.targetModel(nextStates)

            nextQValues[nextValidMoves == False] = float("-inf")

            bestNextQValues = nextQValues.max(dim=1).values

            bestNextQValues[dones == 1] = 0

            targetQValues = (
                rewards
                + self.gamma * bestNextQValues
            )

        loss = self.lossFunction(
        currentQValues,
        targetQValues
        )
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def updateTargetModel(self):
        self.targetModel.load_state_dict(
        self.model.state_dict()
    )   

    def reduceEpsilon(self):
        if self.epsilon > self.epsilonMinimum:
            self.epsilon *= self.epsilonDecay

        if self.epsilon < self.epsilonMinimum:
            self.epsilon = self.epsilonMinimum  

    def saveModel(self, filename):
        torch.save({
        "modelState": self.model.state_dict(),
        "targetModelState": self.targetModel.state_dict(),
        "optimizerState": self.optimizer.state_dict(),
        "epsilon": self.epsilon
    }, filename)


    def loadModel(self, filename):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(
            checkpoint["modelState"]
        )

        self.targetModel.load_state_dict(
            checkpoint["targetModelState"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizerState"]
        )

        self.epsilon = checkpoint["epsilon"]

        print(f"Modellen ble lastet fra {filename}")

    def preprocessState(self, board):
        board = np.asarray(board, dtype=np.float32)

        processedState = np.zeros_like(board, dtype=np.float32)
        occupied = board > 0
        processedState[occupied] = np.log2(board[occupied])

        return processedState.flatten()

    
