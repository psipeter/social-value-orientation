import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import nni

from game import *
from utils import *
from agents import *
from T4T import *

def run_greedy_trustee(agents, nGames, verbose=False, train=True):
    dfs = []
    for a, agent in enumerate(agents):
        if verbose: print(f"{agent.ID}")
        seed = a if train else 1000+a
        t4ts = make_greedy_trustees(nGames, seed=seed)
        for g in range(nGames):
            if verbose: print(f"game {g}")
            df = play_game(agent, t4ts[g], gameID=g, train=train)
            dfs.extend(df)
        del(agent)
    data = pd.concat(dfs, ignore_index=True)
    return data

def main(args):
    architecture = args['architecture']
    nAgents = args['nAgents']
    nGames_train = args['nGames_train']
    nGames_test = args['nGames_test']

    agents = []
    for n in range(nAgents):
        if architecture=='DQN':
            agents.append(
                DQN('investor',
                    ID=f"DQN{n}",
                    seed=n,
                    nActions=args['nActions'],
                    nNeurons=args['nNeurons'],
                    tau=args['tau'],
                    alpha=args['alpha'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    update=args['update'],
                    nGames=nGames_train,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
        elif architecture=='IBL':
            agents.append(
                IBL('investor',
                    ID=f"IBL{n}",
                    seed=n,
                    nActions=args['nActions'],
                    tau=args['tau'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    nGames=nGames_train,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
        elif architecture=='NEF':
            agents.append(
                NEF('investor',
                    ID=f"NEF{n}",
                    seed=n,
                    nEns=args['nEns'],
                    nArr=args['nArr'],
                    nStates=args['nStates'],
                    nActions=args['nActions'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    nGames=nGames_train,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))

    nni.report_intermediate_result(0)
    agentIDs = [agent.ID for agent in agents]
    data_train = run_greedy_trustee(agents, nGames=nGames_train, train=True, verbose=True).query("ID in @agentIDs")
    data_test = run_greedy_trustee(agents, nGames=nGames_test, train=False, verbose=True).query("ID in @agentIDs")
    score = data_test['coins'].mean()
    # score = data_test.query("ID in @agentIDs")['coins'].to_numpy()
    # print(score)

    # nni.report_intermediate_result(score)
    nni.report_final_result(score)

if __name__ == '__main__':
    params = {
        'architecture': 'NEF',
        'nAgents': 1,
        'nGames_train': 2,
        'nGames_test': 2,
        'nActions': 11,
        'nEns': 300,
        'nArr': 100,
        'nStates': 100,
        'nNeurons': 30,
        'tau': 3,
        'alpha': 0.1,
        'gamma': 0.9,
        'explore': 'linear',
        'update': 'SARSA',
        'w_s': 1.0,
        'w_o': 0.0,
        'w_i': 0.0,
    }
    # optimized_params = nni.get_next_parameter()
    # params.update(optimized_params)
    main(params)

# "w_o": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# "w_i": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# "explore": {"_type":"choice","_value":["linear", "exponential"]},
# "tau": {"_type":"quniform","_value":[1.0, 20.0, 0.1]},
# "nNeurons": {"_type":"randint","_value":[20, 100]},
# "update": {"_type":"choice","_value":["Q-learning", "SARSA"]},
