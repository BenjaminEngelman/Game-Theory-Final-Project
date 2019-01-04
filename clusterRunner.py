from helper import parallelize, saveComplexJson
from agent import  AgentWithSingleAlgo, AgentWithEnsemble
from mazes import *
from experimentConf import *
import sys

if len(sys.argv) < 3:
    print("Not enough arguments, pass either 'single' or 'ensemble and the experiment number'")
    exit(0)

mode = sys.argv[1]
if mode not in ['single', 'ensemble']:
    print("Invalid mode, pass either 'single' or 'ensemble'")
    exit(0)

expNum = int(sys.argv[2])
if expNum not in [3, 4, 5]:
    print("Invalid experiment, pass either 3, 4 or 5")
    exit(0)

resultsFilename = "%s-%d.json" % (mode, expNum)
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

print("Saving results of experiment %d in %s mode in filename %s" % (expNum, mode, resultsFilename))



def runTrialSingleAlgorithm(algorithm, params, numSteps, maze):
    agent = AgentWithSingleAlgo(maze, algorithm, params)

    # reward intake = reward moyen par mouvement
    # Il mesure deux choses
    # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
    allRewardIntakes, numberOfSteps = agent.learn(numSteps)
    return allRewardIntakes, numberOfSteps

def runTrialEnsemble(ensemble, algoParams, temp, numSteps, maze):
    agent = AgentWithEnsemble(maze, ensemble, algoParams, temp, neural=True)
    allRewardIntakes, numberOfSteps = agent.learn(numSteps)
    return allRewardIntakes, numberOfSteps



def addJobsSingleAlgorithm(jobs, pool):
    numSteps = 20000
    for i in range(500):
        maze = mazeGenerator()
        for algorithmName, algorithm, algoParams in algos:
            jobs[(algorithmName, i)] = pool.apply_async(runTrialSingleAlgorithm, (algorithm, algoParams, numSteps, maze))

def addJobsEnsemble(jobs, pool):
    numSteps = 50000
    for i in range(500):
        maze = mazeGenerator()
        for ensembleName, ensemble, algoParams, temp in ensembles:
            jobs[(ensembleName, i)] = pool.apply_async(runTrialEnsemble, (ensemble, algoParams, temp, numSteps, maze))



#def jobDone(key):
#    algorithmName, i  = key
#    print("Done for algorithm %s (iteration %d)" % (algorithmName, i))

if mode == 'single':
    results = parallelize(addJobsSingleAlgorithm, numProcesses=32)
    saveComplexJson(resultsFilename, results)
elif mode == 'ensemble':
    results = parallelize(addJobsEnsemble, numProcesses=32)
    saveComplexJson(resultsFilename, results)