import numpy as np
import random
import pandas as pd
import time
from utils import *

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

def play_game(investor, trustee, gameID, dfs, train):
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
		investor.learn(game)
		trustee.learn(game)
	return dfs

def run(investors, trustees, player, nGames, verbose=False, train=True):
	agents = investors if player=='investor' else trustees
	opponents = trustees if player=='investor' else investors
	dfs = []
	for agent in agents:
		print(f"{agent.ID} vs {opponents[0].ID}")
		for g in range(nGames):
			start_time = time.time()
			if player=='investor':
				dfs = play_game(agent, trustees[g], gameID=g, dfs=dfs, train=train)
			elif player=='trustee':
				dfs = play_game(investors[g], agent, gameID=g, dfs=dfs, train=train)
			end_time = time.time()
			if verbose: print(f"game {g}, execution time {end_time-start_time:.3}")
		del(agent)
	data = pd.concat([df for df in dfs], ignore_index=True)
	return data
