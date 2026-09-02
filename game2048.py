import numpy as np
import random as random


class game2048:

    def __init__(self):
        self.board = np.zeros((4, 4), dtype = int)
        self.score = 0

        # sette opp brett med to tilfeldig plasserte brikker
        for i in range(2):
            while True: 
                indeks1 = random.randint(0, 3)
                indeks2 = random.randint(0, 3)

                if self.board[indeks1, indeks2] == 0:
                    break
            if random.random() < 0.9:
                self.board[indeks1, indeks2] = 2
            else:
                self.board[indeks1, indeks2] = 4
        self.gameover = False

        values = 1 ** np.arange(1, 17)
        self.rewardBoard = values.reshape(4,4)

    def reset(self):
        self.score = 0 
        self.gameover = False
        self.board = np.zeros((4, 4), dtype = int)
        for i in range(2):
            while True: 
                indeks1 = random.randint(0, 3)
                indeks2 = random.randint(0, 3)

                if self.board[indeks1, indeks2] == 0:
                    break
            if random.random() < 0.9:
                self.board[indeks1, indeks2] = 2
            else:
                self.board[indeks1, indeks2] = 4


    def printBoard(self):
        print(self.board)

    def moveLeft(self):

        # Flytte brikker til venstre med logikke fra det orginale spillet
        for i in range(3, -1, -1):
            row = self.board[i].tolist()
            for k in range(3, -1, -1):
                if row[k] == 0:
                    row.pop(k)
                    row.append(0)
            for k in range(0, 3):
                if row[k] == row[k+1]:
                    row[k] = 2 * row[k]
                    self.score += row[k]
                    row.pop(k+1)
                    row.append(0)
            self.board[i] = row

        


                    

    def moveRight(self):

        # Snu brettet left-to-right så flyttet til venstre og snu brettet tilbake
        self.board = np.fliplr(self.board)
        self.moveLeft()
        self.board = np.fliplr(self.board)

    def moveUp(self):

        # Transponerer, flytte, transponerer tilbake
        self.board = self.board.T
        self.moveLeft()
        self.board = self.board.T

    def moveDown(self):
        self.board = self.board.T
        self.moveRight()
        self.board = self.board.T

    def checkGameOver(self):

        # Hvis noen av brikken er null
        if np.any(self.board == 0):
            return False

        # Like naboer horisontalt
        if np.any(self.board[:, : -1] == self.board[:, 1:]):
            return False

        # Like naboer vertikalt
        if np.any(self.board[:-1, :] == self.board[1: , :]):
            return False

        else:
            return True 
        
    def addTile(self):
        while True: 
            indeks1 = random.randint(0, 3)
            indeks2 = random.randint(0, 3)

            if self.board[indeks1, indeks2] == 0:
                break
        if random.random() < 0.9:
            self.board[indeks1, indeks2] = 2
        else:
            self.board[indeks1, indeks2] = 4
            

    def getValidMoves(self):
        slideUp = False
        slideDown = False
        slideRight = False
        slideLeft = False

        originalScore = self.score
        originalBoard = self.board.copy()

        # Sjekker om brettet endrer seg når du utførere en action
        self.moveUp()
        if not np.array_equal(self.board, originalBoard):
            slideUp = True

        self.board = originalBoard.copy()
        self.moveDown()
        if not np.array_equal(self.board, originalBoard):
            slideDown = True

        self.board = originalBoard.copy()
        self.moveLeft()
        if not np.array_equal(self.board, originalBoard):
            slideLeft = True

        self.board = originalBoard.copy()
        self.moveRight()
        if not np.array_equal(self.board, originalBoard):
            slideRight = True

        self.board = originalBoard.copy()
        self.score = originalScore
        return slideRight, slideLeft, slideDown, slideUp      

    def step(self, action):
        oldScore = self.score
        oldBoard = self.board.copy()

        if action == 0:
            self.moveRight()

        if action == 1:
            self.moveLeft()

        if action == 2:
            self.moveDown()

        if action == 3:
            self.moveUp()

        self.gameover = self.checkGameOver()

        validMove = not np.array_equal(self.board, oldBoard)

        if validMove:
            self.addTile()

        reward = self.score - oldScore + np.sum(self.board * self.rewardBoard)

        

        return self.board, reward, self.gameover, validMove








