import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import nni

from game import *
from utils import *
from plots import *
from agents import *
from T4T import *

def main(args):
    architecture = args['architecture']
    nAgents = args['nAgents']
    nGames_train = args['nGames_train']
    nGames_test = args['nGames_test']
    opponent = args['opponent']

    if opponent=='greedy trustee':
        t4ts_train = make_greedy_trustees(nGames_train, seed=0)
        t4ts_test = make_greedy_trustees(nGames_test, seed=1)
    if opponent=='generous investor':
        t4ts_train = make_generous_investors(nGames_train, seed=0)
        t4ts_test = make_generous_investors(nGames_test, seed=1)


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
                    nGames=nGames_train,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i'],
                    representation=args['representation']))

    agentIDs = [agent.ID for agent in agents]
    data_train = run(agents, t4ts_train, 'investor', nGames=nGames_train, train=True)
    data_test = run(agents, t4ts_test, 'investor', nGames=nGames_test, train=False)
    score = data_test.query("ID in @agentIDs")['coins'].mean()
    # score = data_test.query("ID in @agentIDs")['coins'].to_numpy()
    # print(score)

    nni.report_intermediate_result(score)
    nni.report_final_result(score)

if __name__ == '__main__':
    params = {
        'architecture': 'DQN',
        'opponent': 'greedy trustee',
        'nAgents': 10,
        'nGames_train': 100,
        'nGames_test': 10,
        'nActions': 11,
        'nNeurons': 30,
        'tau': 3,
        'alpha': 0.1,
        'gamma': 0.9,
        'explore': 'exponential',
        'w_s': 1.0,
        'w_o': 0.0,
        'w_i': 0.0,
        'representation': 'one-hot',
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    main(params)