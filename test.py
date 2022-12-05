from agents import *
from game import *
from T4T import *
import matplotlib.pyplot as plt
import seaborn as sns
import time
palette = sns.color_palette("colorblind")
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})

nGames = 150
nAgents = 1
player = 'investor'
nActions = 11 if player=='investor' else 31
player2 = 'trustee' if player=='investor' else 'investor'
opponent = 'greedy'
w_o = 0.0
w_i = 0.0
# agents = [DQN(player=player, nActions=nActions, nGames=nGames, explore='linear', w_o=w_o, w_i=w_i)]
# agents = [IBL(player=player, nActions=nActions, nGames=nGames, explore='linear', w_o=w_o, w_i=w_i, decay=0.5, sigma=0.1, thrA=-1.5, seed=1)]
agents = [NEF(player=player, nActions=nActions, nGames=nGames, seed=n, ID="NEF"+str(n), update="Q-learning",
	nNeuronsState=10000, nStates=155, alpha=1e-7) for n in range(nAgents)]
IDs = [agent.ID for agent in agents]

data = run(agents, nGames=nGames, opponent=opponent, train=True, verbose=True).query("ID in @IDs")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=((6.38, 3)))
sns.histplot(data=data, x='game', y='generosity', bins=(15, 11), binrange=((0, nGames),(0, 1)), ax=ax, color=palette[0])
ax2 = ax.twinx()
sns.lineplot(data=data, x='game', y='coins', ax=ax2, color=palette[1])
ax.set(title=f'{IDs[0]} vs {opponent} {player2}', xlabel='Game', ylabel='Generosity', yticks=((0.0, 0.2, 0.4, 0.6, 0.8, 1.0)))
ax2.set(ylabel='Score', yticks=((5,10,15)))
plt.tight_layout()
fig.savefig(f"plots/test_{IDs[0]}_{opponent}_{player2}.png", dpi=600)