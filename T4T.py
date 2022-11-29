import numpy as np
import random
from utils import *

class T4T():
	def __init__(self, player, o=1, f=1, p=1, x=0.5, c=0.2, ID="T4T"):
		self.player = player
		self.ID = ID
		self.o = o  # initial state of the agent
		self.x = x  # expected generosity of opponent (fraction of capital given, fraction of available money returned)
		self.f = f  # rate of forgiveness (state increase with opponent generosity)
		self.p = p  # rate of punishment (state decrease with opponent greed)
		self.c = c  # comeback rate (state change if opponent had a forced skip last turn)
		self.state = self.o if player=="investor" else self.o/2 # dynamic agent state

	def new_game(self, game):
		self.state = self.o if self.player=="investor" else self.o/2

	def move(self, game):
		self.update_state(game)
		coins = game.coins if self.player=='investor' else game.giveI[-1]*game.match  # coins available on this turn
		action = coins * self.state
		give = int(np.clip(action, 0, coins))
		keep = int(coins - give)
		return give, keep

	def update_state(self, game):
		if self.player == "investor":
			if len(game.giveT)==0: return
			g = game.genT[-1]
			if np.isnan(g):
				# if opponent was skipped last turn, agent state goes from zero to self.C (*self.F)
				delta = self.c
			else:
				# delta proportional to generosity fraction minus expected generosity (self.X)
				delta = g - self.x
		else:
			g = game.genI[-1]
			# delta proportional to generosity fraction minus expected generosity (self.X)
			delta = g - self.x
		if delta > 0:
			self.state += delta*self.f
		elif delta < 0:
			self.state += delta*self.p
		if self.player=='investor':
			self.state = np.clip(self.state, 0, 1.0)
		else:
			self.state = np.clip(self.state, 0, 0.5)

	def learn(self, game):
		pass

def make_greedy_investors(games, seed=0):
	rng = np.random.RandomState(seed=seed)
	os = rng.uniform(0.8, 1.0, size=games)
	xs = rng.uniform(0.5, 0.5, size=games)
	fs = rng.uniform(1.0, 1.0, size=games)
	ps = rng.uniform(0.1, 0.3, size=games)
	T4Ts = [T4T("investor", o=os[g], x=xs[g], f=fs[g], p=ps[g], ID=f"T4T{g}") for g in range(games)]
	return T4Ts

def make_greedy_trustees(games, seed=0):
	rng = np.random.RandomState(seed=seed)
	os = rng.uniform(0.1, 0.3, size=games)
	xs = rng.uniform(0.5, 0.5, size=games)
	fs = rng.uniform(0.0, 0.1, size=games)
	ps = rng.uniform(0.2, 0.2, size=games)
	T4Ts = [T4T("trustee", o=os[g], x=xs[g], f=fs[g], p=ps[g], ID=f"T4T{g}") for g in range(games)]
	return T4Ts

def make_generous_investors(games, seed=0):
	rng = np.random.RandomState(seed=seed)
	os = rng.uniform(0.6, 0.8, size=games)
	xs = rng.uniform(0.5, 0.5, size=games)
	fs = rng.uniform(0.8, 1.0, size=games)
	ps = rng.uniform(1.0, 1.0, size=games)
	T4Ts = [T4T("investor", o=os[g], x=xs[g], f=fs[g], p=ps[g], ID=f"T4T{g}") for g in range(games)]
	return T4Ts

def make_generous_trustees(games, seed=0):
	rng = np.random.RandomState(seed=seed)
	os = rng.uniform(0.3, 0.5, size=games)
	xs = rng.uniform(0.5, 0.5, size=games)
	fs = rng.uniform(0.4, 0.6, size=games)
	ps = rng.uniform(1.0, 1.0, size=games)
	T4Ts = [T4T("trustee", o=os[g], x=xs[g], f=fs[g], p=ps[g], ID=f"T4T{g}") for g in range(games)]
	return T4Ts