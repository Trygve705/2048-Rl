from agent import agent 
from game2048 import game2048
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from test import testModell


agent = agent()
agent.loadModel("bestModelForbedret.pth")
agent.epsilon = 0.5

numberOfEpisodes = 100_000
trainingStep = 0
score = []
runde = []
maxTile = []

    

for episode in range(numberOfEpisodes):

    game = game2048()

    state = game.board.copy()
    done = False
    

    while not done:

        validMoves = game.getValidMoves()

        if not any(validMoves):
            done = True
            break


        action = agent.chooseAction(state, validMoves)

        nextState, reward, done, validMove = game.step(action)

        nextValidMoves = game.getValidMoves()

        agent.remember(
            state,
            action,
            reward,
            nextState,
            done,
            nextValidMoves
        )
        if trainingStep % 4 == 0:
            loss = agent.learn()

        state = nextState.copy()

        trainingStep += 1
        
        if trainingStep % 1000 == 0:
            agent.updateTargetModel()

        
         
    if (episode + 1) % 30_000 == 0:
        # Lagrer epsilon
        epsilonMid = agent.epsilon

        if testModell("bestModelForbedret.V2.pth", agent):
            agent.saveModel("bestModelForbedret.V2.pth")
            print("Ny modell lagret")

        else:
            agent.loadModel("bestModelForbedret.V2.pth")
            agent.updateTargetModel()
            print("Lagret ikke ny modell")

        # Setter tilbake epislon
        agent.epsilon= epsilonMid

    agent.reduceEpsilon()

    print(f"Runde: {episode}, Score: {game.score}, Episolon: {agent.epsilon}")

    score.append(game.score)
    runde.append(episode + 1)
    maxTile.append(np.max(game.board))


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

averageScores = pd.Series(score).rolling(200).mean()

# Venstre plot: score
ax[0].plot(runde, score, color="red", alpha=0.3, label="Score")
ax[0].plot(runde, averageScores, linewidth=2, label="Snitt siste 200")
ax[0].set_title("Score")
ax[0].set_xlabel("Runde")
ax[0].set_ylabel("Score")
ax[0].legend()

# Høyre plot: høyeste brikke
ax[1].plot(runde, maxTile, color="blue")
ax[1].set_title("Høyeste brikke")
ax[1].set_xlabel("Runde")
ax[1].set_ylabel("Brikkeverdi")

plt.tight_layout()
plt.show()