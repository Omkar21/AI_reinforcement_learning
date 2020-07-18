# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qVal= util.Counter();
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #getting the Q value for particular states and actions
        qvalues= self.qVal[(state, action)]
        return qvalues
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
         #getting all the legal actions
        lenght = len(self.getLegalActions(state))
        if lenght:
          True
        else:
            x=0
            return x
          # initiatinng a list for qvalues od the state
        statQval = util.Counter()
        for A in self.getLegalActions(state):
            #fillling all the q values in the empty list
          statQval[A] = self.getQValue(state,A)
          z= statQval.argMax()
          stateqvalues = statQval[z]
            #return the larger of the q values
        return stateqvalues
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #initiating a variable with best value as -ve infinity and best action as none
        bval = -999999999999999
        bact = None
        #getting all the legal actions
        #running a loop for all the legal actions
        for A in self.getLegalActions(state):
          #check if the q value obtained is greater then best value change best value to the q value obtained
          if bval > self.getQValue(state, A):
              #if best best value is greater than the obtained value then we pass it
            pass
          elif bval<self.getQValue(state, A):
              #if best value is less than the qvalue obtained then we change the best value to q value
              temp = self.getQValue(state, A)
              bval = temp
              # assign the corresponding action with it to the best action
              bact = A
          elif bval==self.getQValue(state, A):
              # if the best value is equalu to the qvalue obtained then we change the vest value to qvalue
              temp=self.getQValue(state, A)
              bval = temp
            #assign the corresponding action with it to the best action
              bact = A
        return bact

        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        y = self.epsilon
        x = util.flipCoin(y)
        #     #selection actions from q value if it is not flip coin probability
        action = self.computeActionFromQValues(state) if not x else random.choice(legal)
        #   #selection action ramdomly if it is probability of flip coin


        return action
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #start calculating the sample value according the equation given
        disc, next = self.discount , self.getValue(nextState)
        #getting the value of alpha
        #getting the q values calculates according the equation
        z = (1 - self.alpha)*self.qVal[(state,action)]
        y = self.alpha*(reward + disc*next)
        self.qVal[(state,action)] = z + y
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #getting the features
        qval = 1
        for F in self.featExtractor.getFeatures(state, action):
          qval += self.weights[F]*self.featExtractor.getFeatures(state, action)[F]
        return qval-1
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #calculating the sample and update according to the equation
        disc = self.discount
        qv = self.computeValueFromQValues(nextState)
        diff = (reward + disc*qv) - self.getQValue(state,action)
        for F in self.featExtractor.getFeatures(state, action):
            alpha = self.alpha
            self.weights[F] += alpha*diff*self.featExtractor.getFeatures(state, action)[F]
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print (self.update(state))
