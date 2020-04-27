from __future__ import division

import time
import math
import random
import copy
import torch
from torch_geometric.data import Data
import numpy as np
from TSP import Action

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode():
    def __init__(self, state, parent,t,r,network):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}
        self.tour=t
        self.graph=None
        self.tsp=network
        self.action=self.tour[-1]
        self.remain=r if r is not [] else list(range(1, state.num_nodes))


    def construct_graph(self):
        if self.graph is not None:
            return self.graph

        points = torch.tensor(self.tsp.pos).to(dtype=torch.float)

        edges = torch.zeros((2, len(self.tour) - 1), dtype=torch.long)
        for i in range(len(self.tour) - 1):
            edges[0, i] = self.tour[i]
            edges[1, i] = self.tour[i + 1]

        choices = torch.zeros(self.state.num_nodes, dtype=torch.bool)
        choices[self.remain] = 1
        x = torch.cat([points, choices.unsqueeze(-1).to(dtype=torch.float)], dim=-1)

        self.graph = Data(x=x, pos=points, edge_index=edges, y=choices)

        return self.graph


class mcts():
    #explorationConstant=75 for best
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=75,
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState,taken_tour,parent):
        node_list=list(range(0,initialState.num_nodes))
        remain=[i for i in node_list if i not in taken_tour]
        self.root = treeNode(initialState, parent,taken_tour,remain,initialState.network)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return bestChild,self.getAction(self.root, bestChild)

    def search_policy(self, initialState,model,taken_tour,parent):
        node_list = list(range(0, initialState.num_nodes))
        remain = [i for i in node_list if i not in taken_tour]
        self.root = treeNode(initialState, parent, taken_tour, remain, initialState.network)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound_policy(model)
        else:
            for i in range(self.searchLimit):
                self.executeRound_policy(model)

        bestChild = self.getBestChild(self.root, 0)
        return bestChild,self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def executeRound_policy(self,model):
        node = self.selectNode_policy(self.root,model)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        random.shuffle(actions)
        for action in actions:
            if action.get_tuple() not in node.children.keys():
                new_tour=copy.copy(node.tour)
                new_tour.append(action.y)
                new_remain=copy.copy(node.remain)
                if new_remain!=[]:
                    try:
                        new_remain.remove(action.y)
                    except:
                        print('error')
                newNode = treeNode(node.state.takeAction(action), node,new_tour,new_remain,node.state.network)
                node.children[action.get_tuple()] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def selectNode_policy(self,node,model):

        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild_policy(node, model)
            else:
                return self.expand(node)
        return node

    def getBestChild_policy(self,node,model):
        if len(node.children)==1: return node.children[next(iter(node.children))]

        model.eval()

        actions = [val.action for key,val in node.children.items()]
        r = list(set.intersection(set(actions), set(node.remain)))
        z = np.zeros(node.state.num_nodes, dtype=np.int)
        z[r] = 1
        z = z[node.remain]

        graph = node.construct_graph()
        pred, value = model(graph)

        pred = pred.squeeze()[z]
        selection = torch.multinomial(pred, 1).tolist()[0]

        act=Action(1,node.action,node.remain[selection])
        return node.children[act.get_tuple()]

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return Action(1,action[0],action[1])