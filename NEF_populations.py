import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from statannotations.Annotator import Annotator
from matplotlib.ticker import FormatStrFormatter

from game import *
from utils import *
from plots import *
from agents import *
from T4T import *
from nni_pop import *

palette = sns.color_palette("colorblind")
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})

def rerun(agents, args, t4tSeedStart=0):
    dfs = []
    for i in range(args['nTest']):
        print(f'test {i}')
        for agent in agents:
            agent.reinitialize(args['player'])
        df = run(agents, nGames=args['nGames'], opponent=args["opponent"], t4tSeed=t4tSeedStart+i, verbose=True).query("ID in @IDs")
        df['t4tSeed'] = [t4tSeedStart+i for _ in range(df.shape[0])]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

def tTestAndPlot(emp, sim, args, dependent):
    player = args['player']
    opponent = args['opponent']
    gameFinal = 14 - args['nFinal']
    player2 = 'trustee' if player=='investor' else 'investor'
    emp = emp.query('game>@gameFinal')
    sim = sim.query('game>@gameFinal')
    yticks = ((0, 0.2, 0.4, 0.6, 0.8, 1.0)) if dependent=='generosity' else ((0, 5, 10, 15))

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=((6.38, 2)))
    sns.violinplot(data=emp, x="orientation", y=dependent, order=["proself", "prosocial"], ax=axes[0], palette=palette[2:4], saturation=1, inner='quartile', cut=0, bw=0.2)
    annot = Annotator(pairs=[("proself", "prosocial")], data=emp, x='orientation', y=dependent, order=["proself", "prosocial"], ax=axes[0], plot='violinplot')
    annot.configure(test="t-test_ind", loc='inside', verbose=1).apply_test().annotate()
    axes[0].set(xlabel=None, title=f'humans vs {opponent} {player2}', yticks=yticks)

    sns.violinplot(data=sim, x="orientation", y=dependent, order=["proself", "prosocial"], ax=axes[1], palette=palette[2:4], saturation=1, inner='quartile', cut=0, bw=0.2)
    annot = Annotator(pairs=[("proself", "prosocial")], data=sim, x='orientation', y=dependent, order=["proself", "prosocial"], ax=axes[1], plot='violinplot')
    annot.configure(test="t-test_ind", loc='inside', verbose=1).apply_test().annotate()
    axes[1].set(xlabel=None, ylabel=None, title=f'agent vs {opponent} {player2}', yticks=yticks)

    sns.despine(fig=fig, ax=axes, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    plt.tight_layout()
    fig.savefig(f"plots/{args['architecture']}_vs_{opponent}_{player2}_{dependent}.png", dpi=600)

def LTPlot(agents, sim, args):
    player2 = 'trustee' if args['player']=='investor' else 'investor'
    IDs = [agent.ID for agent in agents]
    for ID in IDs:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=((6.38, 2)))
        sns.histplot(data=sim.query("ID==@ID"), x='game', y='generosity', bins=(15, 11), binrange=((0, args['nGames']),(0, 1)), ax=axes, color=palette[0])
        ax2 = axes.twinx()
        sns.lineplot(data=sim.query("ID==@ID"), x='game', y='coins', ax=ax2, color=palette[1])
        axes.set(title=f'{ID} vs {args["opponent"]} trustee',
            xlabel='Game', ylabel='Generosity', xticks=((0,5,10,15)), yticks=((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
        ax2.set(ylabel='Score', yticks=((3,6,9,12,15)))
        plt.tight_layout()
        fig.savefig(f"plots/{ID}_vs_{opponent}_{player2}_LT.png", dpi=600)
        plt.close("all")


params = {
    "architecture": "NEF",
    "player": "investor",
    "opponent": "generous",
    "nAgents": 1,
    "nTrain": 1,
    "nTest": 1,
    "nGames": 15,
    "explore": "exponential",
    "update": "Q-learning",
    "w_s": 1,
    "w_o": 0.3,
    "w_i": 0.3,
    "nFinal": 3,
    "optimize_target": "final",
    "overlap_test": "ks",
    "popSize": 1,
}

params2 = {
    "popSeed": 1,
    "thrSVO": 0.3,
    "tau": 6.5,
    "alpha": 1e-7,
    "gamma": 0.5,
    "nStates": 160,
    "radius": 0.5,
}

args = params | params2
agents = makePopulation(args)
IDs = [agent.ID for agent in agents]
player = args['player']
player2 = 'investor' if player=='trustee' else 'trustee'
opponent = args['opponent']
emp = pd.read_pickle("data/human_data_cleaned.pkl").query('player==@player & opponent==@opponent')

data = rerun(agents, args)

pop, selected = selectLearners(agents, data)
sim, nProself, nProsocial = addLabel(pop, selected, args)
print(f"proself {nProself}, prosocial {nProsocial}")
tTestAndPlot(emp, sim, args, 'generosity')

overlapProself = overlap(emp.query("orientation=='proself'"), sim.query("orientation=='proself'"), args)
overlapProsocial = overlap(emp.query("orientation=='prosocial'"), sim.query("orientation=='prosocial'"), args)
print(overlapProself)
print(overlapProsocial)

LTPlot(pop, sim, args)