# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #getting range of iterations list
        ran = range(iterations)
        iteration = 0
        while iteration in ran:
            Value = self.values.copy()
            states = mdp.getStates()
            for state in states:
                valact= util.Counter()
                #get all the possible actions
                actions = mdp.getPossibleActions(state)
                for action in actions:
                    #getting all possible states and problems
                    problem = mdp.getTransitionStatesAndProbs(state, action)
                    for trans, prob in problem:
                        #getting reward for that state and next state
                        reward = mdp.getReward(state, action, trans)
                        #getting the value for that action
                        valact[action] += prob * (reward + discount * Value[trans])
                        #fetting the maximum value by using argmax
                maxValact = valact[valact.argMax()]
                self.values[state] = maxValact
            iteration+=1
        
                
                

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qval = 1
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            #getting the qvalues by calculation by using the transition states and reward function
            qval += transition[1]*self.mdp.getReward(state, action, transition[0]) +transition[1]*self.discount*self.getValue(transition[0])

        return qval-1
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #getting possible actions
        actions = self.mdp.getPossibleActions(state)
        #getting discount
        discount = self.discount
        while actions:
            Actions = actions
            valact = util.Counter()

            for action in Actions:
                #getting possible transitionstates
	            transition = self.mdp.getTransitionStatesAndProbs(state, action)
                    for State, prob in transition:
                        Z = discount * self.values[State]
                        #getting the value of the action
                        valact[action] += prob * (self.mdp.getReward(state, action, State))+ prob*(Z)
                        value = valact.argMax()

            return value

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
