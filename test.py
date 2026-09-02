import numpy as np
from agent import agent
from game2048 import game2048

def testModell(Beste, ny):
    Testagent = agent()
    Testagent.loadModel(Beste)
    Testagent.epsilon = 0.0

    scoresGammel = []
    scoresNy = []
    numberOfTests = 1000

    for episode in range(numberOfTests):
        game = game2048()
        state = game.board.copy()
        done = False

        while not done:
            validMoves = game.getValidMoves()

            if not any(validMoves):
                break

            action = Testagent.chooseAction(state, validMoves)
            nextState, reward, done, validMove = game.step(action)
            state = nextState.copy()


        scoresGammel.append(game.score)

    
    ny.epsilon = 0.0

    for episode in range(numberOfTests):
        game = game2048()
        state = game.board.copy()
        done = False

        while not done:
            validMoves = game.getValidMoves()

            if not any(validMoves):
                break

            action = ny.chooseAction(state, validMoves)
            nextState, reward, done, validMove = game.step(action)
            state = nextState.copy()


        scoresNy.append(game.score)

    snittNy = sum(scoresNy)/len(scoresNy)
    snittGammel = sum(scoresGammel)/len(scoresGammel)

    if snittNy > (snittGammel * 0.90):
        return True
    else:
        return False