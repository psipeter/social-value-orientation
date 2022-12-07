import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import time

from game import *
from utils import *
from plots import *
from agents import *
from T4T import *

palette = sns.color_palette("colorblind")
sns.set_palette(palette)
sns.set(context='paper', style='white', rc={'font.size':12, 'mathtext.fontset': 'cm'})

player = 'investor'
player2 = 'trustee'
opponent = "greedy"
nGames = 15
nAgents = 25

agents = []
IDs = []
for n in range(nAgents):
    agents.append(
        NEF(
            'investor',
            ID=f"NEF{n}",
            seed=n,
            gamma=0.1,
            explore='linear',
            update='Q-learning',
            representation='ssp',
            radius=0.5,
            nNeuronsState=16000,
            nStates=160,
            alpha=1e-7,
            w_s=1.0,
            w_o=0.0,
            w_i=0.0,
            nGames=nGames))
    IDs.append(agents[-1].ID)

data = run(agents, nGames=nGames, opponent='greedy', train=True, verbose=True).query("ID in @IDs")
data.to_pickle(f"data/NEF_vs_{opponent}_{player2}_{nGames}games.pkl")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((6.38, 3)))
sns.histplot(data=data, x='game', y='generosity', bins=(15, 11), binrange=((0, nGames),(0, 1)), ax=ax, color=palette[0])
ax2 = ax.twinx()
sns.lineplot(data=data, x='game', y='coins', ax=ax2, color=palette[1])
ax.set(title='NEF vs greedy trustee', xlabel='Game', ylabel='Generosity', yticks=((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
ax2.set(ylabel='Score', yticks=((0, 5, 10)))
plt.tight_layout()
fig.savefig(f"plots/NEF_vs_{opponent}_{player2}_{nGames}games.png", dpi=600)