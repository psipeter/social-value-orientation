import numpy as np
import random
import pandas as pd
import time
from utils import *
from T4T import *
from agents import NEF

class Game():
	def __init__(self, coins=10, match=3, turns=5, train=True):
		self.coins = coins
		self.match = match
		self.turns = turns
		self.giveI = []
		self.keepI = []
		self.genI = []
		self.rI = []
		self.giveT = []
		self.keepT = []
		self.genT = []
		self.rT = []
		self.train = train

def play_game(investor, trustee, gameID, train):
	dfs = []
	columns = ('ID', 'opponent', 'player', 'game', 'turn', 'generosity', 'coins')
	game = Game(train=train)
	investor.new_game(game)
	trustee.new_game(game)
	assert investor.player == 'investor' and trustee.player == 'trustee'
	for t in range(game.turns):
		giveI, keepI = investor.move(game)
		game.giveI.append(giveI)
		game.keepI.append(keepI)
		game.genI.append(generosity('investor', giveI, keepI))
		giveT, keepT = trustee.move(game)
		game.giveT.append(giveT)
		game.keepT.append(keepT)
		game.genT.append(generosity('trustee', giveT, keepT))
		game.rI.append(keepI+giveT)
		game.rT.append(keepT)
		dfs.append(pd.DataFrame([[investor.ID, trustee.ID, 'investor', gameID, t, game.genI[t], game.rI[t]]], columns=columns))
		dfs.append(pd.DataFrame([[trustee.ID, investor.ID, 'trustee', gameID, t, game.genT[t], game.rT[t]]], columns=columns))		
	if game.train:
		if isinstance(investor, NEF) or isinstance(trustee, NEF):  # extra turn for nengo learning
			_, _ = investor.move(game)
			_, _ = trustee.move(game)
		investor.learn(game)
		trustee.learn(game)
	return dfs

def run(agents, nGames, opponent, verbose=False, train=True, t4tSeed=0):
    dfs = []
    player = agents[0].player
    for a, agent in enumerate(agents):
        if verbose: print(f"{agent.ID}")
        if player=='investor' and opponent=='greedy':
            t4ts = make_greedy_trustees(nGames, seed=t4tSeed+a)
        elif player=='investor' and opponent=='generous':
            t4ts = make_generous_trustees(nGames, seed=t4tSeed+a)
        elif player=='investor' and opponent=='test':
            t4ts = make_test_trustees(nGames, seed=t4tSeed+a)
        elif player=='trustee' and opponent=='greedy':
            t4ts = make_greedy_investors(nGames, seed=t4tSeed+a)
        elif player=='trustee' and opponent=='generous':
            t4ts = make_generous_investors(nGames, seed=t4tSeed+a)
        elif player=='trustee' and opponent=='test':
            t4ts = make_test_investors(nGames, seed=t4tSeed+a)
        for g in range(nGames):
            start = time.time()
            if player=='investor':
            	df = play_game(agent, t4ts[g], gameID=g, train=train)
            elif player=='trustee':
            	df = play_game(t4ts[g], agent, gameID=g, train=train)
            end = time.time()
            if verbose: print(f"game {g}, time {end-start:.3}")
            dfs.extend(df)
        del(agent)
    data = pd.concat(dfs, ignore_index=True)
    return data