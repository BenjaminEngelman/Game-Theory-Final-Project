from helper import parallelize, saveComplexJson
from agent import algos, ensembles, AgentWithSingleAlgo, AgentWithEnsemble
from mazes import createSimpleMaze
import sys

if len(sys.argv) < 2:
    print("Not enough arguments, pass either 'single' or 'ensemble'")
    exit(0)

mode = sys.argv[1]
if mode not in ['single', 'ensemble']:
    print("Invalid mode, pass either 'single' or 'ensemble'")
    exit(0)




def runTrialSingleAlgorithm(algorithm, params):
    maze = createSimpleMaze()
    agent = AgentWithSingleAlgo(maze, algorithm, params)

    # reward intake = reward moyen par mouvement
    # Il mesure deux choses
    # 1) Dans 2500 derniers épisodes, fait la moyenne du reward intake
    # 2) Tous les 2500 épisodes, regarde quel est le reward intake, puis à la fin il fait la somme
    final, cumulative = agent.learn(50000)
    return final, cumulative

def runTrialEnsemble(ensemble, algoParams, temp):
    maze = createSimpleMaze()
    agent = AgentWithEnsemble(maze, ensemble, algoParams, temp)
    final, cumulative = agent.learn(50000)
    return final, cumulative



def addJobsSingleAlgorithm(jobs, pool):
    for i in range(500):
        for algorithmName, algorithm, algoParams in algos:
            jobs[(algorithmName, i)] = pool.apply_async(runTrialSingleAlgorithm, (algorithm, algoParams))

def addJobsEnsemble(jobs, pool):
    for i in range(500):
        for ensembleName, ensemble, algoParams, temp in ensembles:
            jobs[(ensembleName, i)] = pool.apply_async(runTrialEnsemble, (ensemble, algoParams, temp))



#def jobDone(key):
#    algorithmName, i  = key
#    print("Done for algorithm %s (iteration %d)" % (algorithmName, i))

if mode == 'single':
    results = parallelize(addJobsSingleAlgorithm, numProcesses=32)
    saveComplexJson("resultsSingleAlgorithm.json", results)
elif mode == 'ensemble':
    results = parallelize(addJobsEnsemble, numProcesses=32)
    saveComplexJson("resultsEnsemble.json", results)