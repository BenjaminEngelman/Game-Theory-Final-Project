from algos import *
from ensembleMethods import *

class AlgoParams():
    def __init__(self, alpha=None, beta=None, gamma=None, temp=None, numHiddenNodes=None, makeNNInput=None, nnInputSize=None):
        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.TEMP = temp
        self.NUM_HIDDEN_NODES = numHiddenNodes
        self.MAKE_NN_INPUT = makeNNInput
        self.NN_INPUT_SIZE = nnInputSize

####################################### EXPERIMENT 1 ###########################################

algosExp1 = [
    ("Q-Learning", QLearningNormal, AlgoParams(alpha=0.2, gamma=0.9, temp=1)),
    ("SARSA", SARSA, AlgoParams(alpha=0.2, gamma=0.9, temp=1)),
    ("Actor-Critic", ActorCritic, AlgoParams(alpha=0.1, beta=0.2, gamma=0.95, temp=1)),
    ("QV-Learning", QVLearning, AlgoParams(alpha=0.2, beta=0.2, gamma=0.9, temp=1)),
    ("ACLA", ACLA, AlgoParams(alpha=0.005, beta=0.1, gamma=0.99, temp=1/9))
]
algoParamsListExp1 = [param[2] for param in algosExp1]

ensemblesExp1 = [
    ("Majority", majorityVote, algoParamsListExp1, 1 / 1.6),
    ("Rank", rankVote, algoParamsListExp1, 1 / 0.6),
    ("Boltzmann Multiplication", boltzmannMultVote, algoParamsListExp1, 1/ 0.2),
    ("Boltzmann Addition", boltzmannAddVote, algoParamsListExp1, 1 / 1),
]
################################################################################################

####################################### EXPERIMENT 3 ###########################################
algosExp3 = [
    ("Q-Learning", QLearningNeuronal, AlgoParams(alpha=0.01, gamma=0.95, temp=1, numHiddenNodes=60, makeNNInput=makeNNInput3, nnInputSize=2*54)),
    ("SARSA", SARSA, AlgoParams(alpha=0.01, gamma=0.95, temp=1, numHiddenNodes=60, makeNNInput=makeNNInput3, nnInputSize=2*54)),
    ("Actor-Critic", ActorCritic, AlgoParams(alpha=0.015, beta=0.003, gamma=0.95, temp=1, numHiddenNodes=60, makeNNInput=makeNNInput3, nnInputSize=2*54)),
    ("QV-Learning", QVLearning, AlgoParams(alpha=0.01, beta=0.01, gamma=0.9, temp=1/0.4, numHiddenNodes=60, makeNNInput=makeNNInput3, nnInputSize=2*54)),
    ("ACLA", ACLA, AlgoParams(alpha=0.06, beta=0.002, gamma=0.98, temp=1/6, numHiddenNodes=60, makeNNInput=makeNNInput3, nnInputSize=2*54))
]
algoParamsListExp3 = [param[2] for param in algosExp3]

ensemblesExp3 = [
    ("Majority", majorityVote, algoParamsListExp3, 1 / 2.6),
    ("Rank", rankVote, algoParamsListExp3, 1 / 0.8),
    ("Boltzmann Multiplication", boltzmannMultVote, algoParamsListExp3, 1/ 0.2),
    ("Boltzmann Addition", boltzmannAddVote, algoParamsListExp3, 1 / 1),
    
]
################################################################################################


####################################### EXPERIMENT 4 ###########################################
algosExp4 = [
    ("Q-Learning", QLearningNeuronal, AlgoParams(alpha=0.005, gamma=0.95, temp=1/0.5, numHiddenNodes=20, makeNNInput=makeNNInput4, nnInputSize=2*54)),
    ("SARSA", SARSA, AlgoParams(alpha=0.008, gamma=0.95, temp=1/0.6, numHiddenNodes=20, makeNNInput=makeNNInput4, nnInputSize=2*54)),
    ("Actor-Critic", ActorCritic, AlgoParams(alpha=0.006, beta=0.008, gamma=0.95, temp=1/0.6, numHiddenNodes=20, makeNNInput=makeNNInput4, nnInputSize=2*54)),
    ("QV-Learning", QVLearning, AlgoParams(alpha=0.012, beta=0.004, gamma=0.95, temp=1/0.6, numHiddenNodes=20, makeNNInput=makeNNInput4, nnInputSize=2*54)),
    ("ACLA", ACLA, AlgoParams(alpha=0.06, beta=0.006, gamma=0.98, temp=1/10, numHiddenNodes=20, makeNNInput=makeNNInput4, nnInputSize=2*54))
]
algoParamsListExp4 = [param[2] for param in algosExp4]

ensemblesExp4 = [
    ("Majority", majorityVote, algoParamsListExp3, 1 / 2.4),
    ("Rank", rankVote, algoParamsListExp3, 1 / 1.2),
    ("Boltzmann Multiplication", boltzmannMultVote, algoParamsListExp3, 1 / 0.2),
    ("Boltzmann Addition", boltzmannAddVote, algoParamsListExp3, 1 / 1),
]
################################################################################################



####################################### EXPERIMENT 5 ###########################################
algosExp5 = [
    ("Q-Learning", QLearningNeuronal, AlgoParams(alpha=0.003, gamma=0.95, temp=1 / 0.3, numHiddenNodes=100, makeNNInput=makeNNInput5, nnInputSize=3*54)),
    ("SARSA", SARSA, AlgoParams(alpha=0.003, gamma=0.92, temp=1 / 0.3, numHiddenNodes=100, makeNNInput=makeNNInput5, nnInputSize=3*54)),
    ("Actor-Critic", ActorCritic, AlgoParams(alpha=0.014, beta=0.0015, gamma=0.95, temp=1 / 0.5, numHiddenNodes=100, makeNNInput=makeNNInput5, nnInputSize=3*54)),
    ("QV-Learning", QVLearning, AlgoParams(alpha=0.002, beta=0.001, gamma=0.95, temp=1/0.2, numHiddenNodes=100, makeNNInput=makeNNInput5, nnInputSize=3*54)),
    ("ACLA", ACLA, AlgoParams(alpha=0.1, beta=0.001, gamma=0.98, temp=1/5, numHiddenNodes=100, makeNNInput=makeNNInput5, nnInputSize=3*54))
]
algoParamsListExp5 = [param[2] for param in algosExp5]

ensemblesExp5 = [
    ("Majority", majorityVote, algoParamsListExp3, 1 / 2.4),
    ("Rank", rankVote, algoParamsListExp3, 1 / 1.0),
    ("Boltzmann Multiplication", boltzmannMultVote, algoParamsListExp3, 1/ 0.2),
    ("Boltzmann Addition", boltzmannAddVote, algoParamsListExp3, 1 / 1),
]