import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import nni

from game import *
from utils import *
from agents import *
from T4T import *

def rmse(A, B):
    return np.sqrt(np.mean(np.square(np.array(A)-np.array(B))))

def getBehavioralSimilarity(sim, args, nBins=3, test='rmse'):
    pid = args['participantID']
    emp = pd.read_pickle("data/human_data_cleaned.pkl").query("ID==@pid & player=='investor'")
    lossesGen = []
    lossesScore = []
    for n in range(nBins):
        left = n*int(15/nBins)
        right = (n+1)*int(15/nBins)
        # print(left, right)
        empN = emp.query("@left <= game & game < @right")
        simN = sim.query("@left <= game & game < @right")
        if args['test']=='rmse':
            empGen = np.histogram(empN['generosity'].to_numpy(), bins=11, range=(0, 1), density=True)[0] / 11
            simGen = np.histogram(simN['generosity'].to_numpy(), bins=11, range=(0, 1), density=True)[0] / 11
            lossesGen.append(rmse(empGen, simGen))
        elif args['test']=='ks':
            empGen = empN['generosity'].to_numpy()
            simGen = simN['generosity'].to_numpy()
            lossesGen.append(scipy.stats.ks_2samp(empGen, simGen)[0])
        empScore = empN['coins'].mean() / 15
        simScore = simN['coins'].mean() / 15
        lossesScore.append(rmse(empScore, simScore))
    # print(lossesGen)
    # print(lossesScore)
    return args['wGen']*np.sum(lossesGen) + args['wScore']*np.sum(lossesScore)

def main(args):
    architecture = args['architecture']
    nAgents = args['nAgents']
    nGames = args['nGames']
    pid = args['participantID']
    seed = args['seed']
    opponent = pd.read_pickle("data/human_data_cleaned.pkl").query("ID==@pid")['opponent'].unique()[0]

    agents = []
    for n in range(nAgents):
        if architecture=='DQN':
            agents.append(
                DQN('investor',
                    ID=f"DQN{n}",
                    seed=seed if nAgents==1 else n,
                    nActions=args['nActions'],
                    nNeurons=args['nNeurons'],
                    tau=args['tau'],
                    alpha=args['alpha'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    update=args['update'],
                    nGames=nGames,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
        elif architecture=='IBL':
            agents.append(
                IBL('investor',
                    ID=f"IBL{n}",
                    seed=seed if nAgents==1 else n,
                    nActions=args['nActions'],
                    tau=args['tau'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    decay=args['decay'],
                    sigma=args['sigma'],
                    thrA=args['thrA'],
                    nGames=nGames,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
        elif architecture=='NEF':
            agents.append(
                NEF('investor',
                    ID=f"NEF{n}",
                    seed=seed if nAgents==1 else n,
                    nEns=args['nEns'],
                    nArr=args['nArr'],
                    nStates=args['nStates'],
                    nActions=args['nActions'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    nGames=nGames,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))

    agentIDs = [agent.ID for agent in agents]
    dfs = []
    for a, agent in enumerate(agents):
        if opponent=='greedy':
            t4ts = make_greedy_trustees(nGames, seed=seed+a)
        elif opponent=='generous':
            t4ts = make_generous_trustees(nGames, seed=seed+a)
        for g in range(nGames):
            df = play_game(agent, t4ts[g], gameID=g, train=True)
            dfs.extend(df)
        sim = pd.concat(dfs, ignore_index=True)
        del(agent)
    sim = pd.concat(dfs, ignore_index=True)
    loss = getBehavioralSimilarity(sim, args)
    nni.report_final_result(loss)
    # print(loss)

if __name__ == '__main__':
    params = {
        'architecture': 'IBL',
        'participantID': 'sree',
        'test': 'ks',
        'wGen': 0.5,
        'wScore': 0.5,
        'nAgents': 30,
        'nGames': 15,
        'nActions': 11,
        'seed': 0,
        'explore': 'exponential',
        'update': 'SARSA',
        'w_s': 1.0,
        'w_o': 0.0,
        'w_i': 0.0,
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    main(params)

# IBL
# {
#     "seed": {"_type":"randint","_value":[0, 1000]},
#     "thrA": {"_type":"quniform","_value":[-2.0, 1.0, 0.01]},
#     "decay": {"_type":"quniform","_value":[-1.0, 0.0, 0.01]},
#     "sigma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "tau": {"_type":"quniform","_value":[1.0, 10.0, 0.1]},
#     "gamma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "w_o": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "w_i": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# }

# DQN
# {
#     "nNeurons": {"_type":"randint","_value":[20, 100]},
#     "explore": {"_type":"choice","_value":["linear", "exponential"]},
#     "tau": {"_type":"quniform","_value":[1.0, 20.0, 0.1]},
#     "alpha": {"_type":"qloguniform","_value":[0.001, 1.0, 0.001]},
#     "gamma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "update": {"_type":"choice","_value":["Q-learning", "SARSA"]},
#     "w_o": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "w_i": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# }

# NEF
# {
#     "nEns": {"_type":"randint","_value":[1000, 1000]},
#     "nArr": {"_type":"randint","_value":[300, 300]},
#     "nStates": {"_type":"randint","_value":[100, 100]},
#     "alpha": {"_type":"qloguniform","_value":[1e-8, 1e-6, 1e-8]},
#     "gamma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# }