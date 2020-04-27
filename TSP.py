from __future__ import division

from copy import deepcopy
import numpy as np


class TSPgame():
    def __init__(self, num_nodes,graph,network):
        self.num_nodes = num_nodes
        self.board = np.zeros((num_nodes, num_nodes))
        self.prev_node = 0
        self.currentPlayer = 1
        self.first_node=0
        self.network=network
        self.graph=graph

        self.edge_dict = {}
        for i in range(0,num_nodes):
            for j in range(0,num_nodes):
                if i==j:
                    self.edge_dict[(i, j)] = -float('inf')
                else:
                    self.edge_dict[(i,j)]=-graph[i][j]


    def getPossibleActions(self):

        possibleActions = []
        count=0
        for i in range(0, self.num_nodes):
            if 1 in self.board[:, i]:
                count=count+1

        for i in range(0, self.num_nodes):
            if count<self.num_nodes-1:
                if i==self.first_node:
                    continue
            if 1 not in self.board[:, i] and i != self.prev_node :
                possibleActions.append(Action(player=self.currentPlayer, x=self.prev_node, y=i))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.board[action.x][action.y] = action.player
        newState.prev_node = action.y
        newState.currentPlayer = self.currentPlayer
        return newState

    def isTerminal(self):
        for i in range(0, self.num_nodes):
            if 1 not in self.board[:, i]:
                return False
        return True

    def getReward(self):
        score = 0
        for i in range(0, self.num_nodes):
            v = i
            u = self.board[:, i].argmax()
            score = score - self.graph[u][v]
        return score

class Action():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def get_tuple(self):
        return (self.x,self.y)

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))



