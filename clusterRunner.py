from helper import parallelize, saveComplexJson
from agent import  AgentWithSingleAlgo, AgentWithEnsemble
from mazes import *
from experimentConf import *
import sys

NUM_STEPS = 20000
NUM_TRIALS = 50

if len(sys.argv) < 3:
    print("Not enough arguments, pass either 'single' or 'ensemble and the experiment number'")
    exit(0)

mode = sys.argv[1]
if mode not in ['single', 'ensemble']:
    print("Invalid mode, pass either 'single' or 'ensemble'")
    exit(0)

expNum = sys.argv[2]
if expNum not in ["3", "4", "5"]:
    print("Invalid mode, pass either 3, 4 or 5")
    exit(0)


if expNum == 3:
    algoParams = algoParamsListExp3
    algos = algosExp3
    ensembles = ensemblesExp3
    mazeGenerator = createDynamicObstaclesMaze

elif expNum == 4:
    algoParams = algoParamsListExp4
    algos = algosExp4
    ensembles = ensemblesExp4
    mazeGenerator = createDynamicGoalMaze


elif expNum == 5:
    algoParams = algoParamsListExp5
    algos = algosExp5
    ensembles = ensemblesExp5
    mazeGenerator = createGeneralizedMaze



def runTrialSingleAlgorithm(algorithm, params, numSteps, maze):
    agent = AgentWithSingleAlgo(maze, algorithm, params)
    StepsOverTime, rewardsOverTime = agent.learn(numSteps)
    return StepsOverTime, rewardsOverTime

def runTrialEnsemble(ensemble, algoParams, temp, numSteps, maze):
    agent = AgentWithEnsemble(maze, ensemble, algoParams, temp, neural=True)
    StepsOverTime, rewardsOverTime = agent.learn(numSteps)
    return StepsOverTime, rewardsOverTime



def addJobsSingleAlgorithm(jobs, pool):
    numSteps = NUM_STEPS
    for i in range(NUM_TRIALS):
        maze = mazeGenerator()
        for algorithmName, algorithm, algoParams in algos:
            jobs[(algorithmName, i)] = pool.apply_async(runTrialSingleAlgorithm, (algorithm, algoParams, numSteps, maze))

def addJobsEnsemble(jobs, pool):
    numSteps = NUM_STEPS
    for i in range(NUM_TRIALS):
        maze = mazeGenerator()
        for ensembleName, ensemble, algoParams, temp in ensembles:
            jobs[(ensembleName, i)] = pool.apply_async(runTrialEnsemble, (ensemble, algoParams, temp, numSteps, maze))



#def jobDone(key):
#    algorithmName, i  = key
#    print("Done for algorithm %s (iteration %d)" % (algorithmName, i))

if mode == 'single':
    results = parallelize(addJobsSingleAlgorithm, numProcesses=32)
    saveComplexJson("resultsSingleAlgorithm.json", results)
elif mode == 'ensemble':
    results = parallelize(addJobsEnsemble, numProcesses=32)
    saveComplexJson("resultsEnsemble.json", results)