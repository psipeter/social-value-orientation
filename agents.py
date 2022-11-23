import numpy as np
import random
import os
import torch
import scipy
import nengo
import itertools
from utils import *

class TQ():
	# Tabular Q-learning agent
	def __init__(self, player, ID="TQ", seed=0, nActions=11, nStates=155, w_s=1, w_o=0, w_i=0,
			tau=1, alpha=1, gamma=0.9, explore='linear', update='SARSA', nGames=100, normalize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.nStates = nStates
		self.nActions = nActions
		self.w_s = w_s
		self.w_o = w_o
		self.w_i = w_i
		self.gamma = gamma
		self.alpha = alpha
		self.tau = tau
		self.epsilon = 1
		self.explore = explore
		self.nGames = nGames
		self.normalize = normalize
		self.update = update
		self.reinitialize(self.player)

	def reinitialize(self, player):
		self.player = player
		self.Q = np.zeros((self.nStates, self.nActions))
		self.state_history = []
		self.action_history = []
		self.episode = 0

	def new_game(self, game):
		if self.player=='investor': assert self.nActions==11
		if self.player=='trustee': assert self.nActions==31
		self.state_history.clear()
		self.action_history.clear()
		if self.explore=='linear':
			self.epsilon = 1 - self.episode / self.nGames
		elif self.explore=='exponential':
			self.epsilon = np.exp(-self.tau*self.episode / self.nGames)
		elif game.train==False:
			self.epsilon = 0

	def move(self, game):
		game_state = get_state(self.player, game=game, agent='TQ')
		# Compute action probabilities for the current state
		Q_state = self.Q[game_state]
		# Sample action from q-values in the current state
		doExplore, randAction = self.rng.uniform(0, 1) < self.epsilon, self.rng.randint(self.nActions)
		action = np.argmax(Q_state) if not doExplore else randAction
		# convert action to number of coins given/kept
		available = game.coins if self.player=='investor' else game.giveI[-1]*game.match  # coins available
		give, keep = action, available-action
		# save state and actions for learning
		self.state_history.append(game_state)
		self.action_history.append(action)
		return give, keep

	def learn(self, game):
		self.episode += 1
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		for t in np.arange(game.turns):
			state = self.state_history[t]
			action = self.action_history[t]
			value = self.Q[state, action]
			if t==(game.turns-1):
				next_value = 0
			else:
				next_state = self.state_history[t+1]
				next_action = self.action_history[t+1]
				if self.update=='Q-learning':
					next_value = np.max(self.Q[next_state])
				elif self.update=='SARSA':
					next_value = self.Q[next_state, next_action]
			delta = rewards[t] + self.gamma*next_value - value
			self.Q[state, action] += self.alpha * delta


class DQN():

	class Critic(torch.nn.Module):
		def __init__(self, nNeurons, nStates, nActions):
			torch.nn.Module.__init__(self)
			self.input = torch.nn.Linear(nStates, nNeurons)
			self.hidden = torch.nn.Linear(nNeurons, nNeurons)
			self.output = torch.nn.Linear(nNeurons, nActions)
			self.apply(self.init_params)
		def forward(self, x):
			x = torch.nn.functional.relu(self.input(x))
			x = torch.nn.functional.relu(self.hidden(x))
			x = self.output(x)
			return x
		def init_params(self, m):
			classname = m.__class__.__name__
			if classname.find("Linear") != -1:
				m.weight.data.normal_(0, 1)
				m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))

	def __init__(self, player, seed=0, nStates=156, nActions=11, nNeurons=30, ID="DQN",
			tau=1, alpha=1e-1, gamma=0.9, explore='linear', nGames=100, w_s=1, w_o=0, w_i=0, representation="one-hot", normalize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.normalize = normalize
		self.nStates = nStates
		self.nActions = nActions
		self.nNeurons = nNeurons
		self.representation = representation
		self.gamma = gamma
		self.alpha = alpha
		self.tau = tau
		self.w_s = w_s
		self.w_o = w_o
		self.w_i = w_i
		self.explore = explore
		self.nGames = nGames
		self.reinitialize(player)

	def reinitialize(self, player):
		self.player = player
		torch.manual_seed(self.seed)
		self.critic = self.Critic(self.nNeurons, self.nStates, self.nActions)
		self.optimizer = torch.optim.Adam(self.critic.parameters(), self.alpha)
		self.value_history = []
		self.state_history = []
		self.action_history = []
		self.episode = 0

	def new_game(self, game):
		if self.player=='investor': assert self.nActions==11
		if self.player=='trustee': assert self.nActions==31
		self.value_history.clear()
		self.state_history.clear()
		self.action_history.clear()
		if self.explore=='linear':
			self.epsilon = 1 - self.episode / self.nGames
		if self.explore=='exponential':
			self.epsilon = np.exp(-self.tau*self.episode / self.nGames)
		elif game.train==False:
			self.epsilon = 0

	def move(self, game):
		game_state = get_state(self.player, game, agent="DQN", dim=self.nStates, representation=self.representation)
		# Estimate the value of the current game_state
		values = self.critic(game_state)			
		# Choose and action based on thees values and some exploration strategy
		doExplore, randAction = self.rng.uniform(0, 1) < self.epsilon, torch.LongTensor([self.rng.randint(self.nActions)])[0]
		action = torch.argmax(values) if not doExplore else randAction
		action = action.detach().numpy()
		# translate action into environment-appropriate signal
		available = game.coins if self.player=='investor' else game.giveI[-1]*game.match  # coins available
		give, keep = action, available-action
		# update histories for learning
		# update histories for learning
		self.value_history.append(values[action])
		self.state_history.append(game_state)
		self.action_history.append(action)
		return give, keep

	def learn(self, game):
		self.episode += 1
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		losses = []
		for t in np.arange(game.turns):
			value = self.value_history[t]
			reward = torch.FloatTensor([rewards[t]])
			if t==(game.turns-1):
				next_value = 0
			else:
				next_value = torch.max(self.value_history[t+1])  # SARSA
			delta = reward + self.gamma*next_value - value
			losses.append(delta**2)
		loss = torch.stack(losses).sum()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


class IBL():

	class Chunk():
		def __init__(self, turn, coins, action, reward, value, episode, decay, sigma):
			self.turn = turn
			self.coins = coins
			self.action = action
			self.reward = reward
			self.value = value
			self.triggers = [episode]
			self.decay = decay  # decay rate for activation
			self.sigma = sigma  # gaussian noise added to activation

		def get_activation(self, episode, rng):
			A = 0
			for t in self.triggers:
				A += (episode - t)**(-self.decay)
			return np.log(A) + rng.logistic(loc=0.0, scale=self.sigma)

	def __init__(self, player, ID="IBL", seed=0, nActions=11,
			decay=0.5, sigma=0.3, thrA=0,
			tau=1, alpha=1e-1, gamma=0.9, explore='linear', nGames=100, w_s=1, w_o=0, w_i=0, representation="one-hot", normalize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.normalize = normalize
		self.rng = np.random.RandomState(seed=seed)
		self.nActions = nActions
		self.gamma = gamma
		self.decay = decay
		self.sigma = sigma
		self.w_s = w_s
		self.w_o = w_o
		self.w_i = w_i
		self.thrA = thrA  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.reinitialize()

	def reinitialize(self, player):
		self.player = player
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.state = None
		self.episode = 0

	def new_game(self, game):
		self.working_memory.clear()
		self.learning_memory.clear()
		self.rng.shuffle(self.declarative_memory)

	def move(self, game):
		turn, coins = get_state(self.player, game, "IBL")
		# load chunks from declarative memory into working memory
		self.populate_working_memory(turn, coins, game)
		# select an action (generosity) that immitates the best chunk in working memory
		self.state = self.select_action()
		# create a new chunk for the chosen action, populate with more information in learn()
		new_chunk = self.Chunk(turn, coins, None, None, None, self.episode, self.decay, self.sigma)
		self.learning_memory.append(new_chunk)
		# translate action into environment-appropriate signal
		give, keep, action_idx = action_to_coins(self.player, self.state, self.nActions, game)
		return give, keep

	def populate_working_memory(self, turn, coins, game):
		self.working_memory.clear()
		for chunk in self.declarative_memory:
			A = chunk.get_activation(self.episode, self.rng)
			S = 1 if turn==chunk.turn and coins==chunk.coins else 0
			if A > self.thrA and S > 0:
				self.working_memory.append(chunk)

	def select_action(self):
		if len(self.working_memory)==0:
			# if there are no chunks in working memory, select a random action
			choice = self.rng.randint(0, self.nActions) / (self.nActions-1)
		elif doExplore:
			choice = randAction
		else:
			# choose an action based on the activation, similarity, reward, and/or value of chunks in working memory
			actions = {}
			for action in np.arange(self.nActions)/(self.nActions-1):
				actions[action] = {'activations':[], 'rewards':[], 'values': [], 'blended': None}
			for chunk in self.working_memory:
				if chunk.action not in actions:
					actions[chunk.action] = {'activations':[], 'rewards':[], 'values': [], 'blended': None}
				actions[chunk.action]['activations'].append(chunk.get_activation(self.episode, self.rng))
				actions[chunk.action]['rewards'].append(chunk.reward)
				actions[chunk.action]['values'].append(chunk.value)
			# compute the blended value for each potential action as the sum of values weighted by activation
			for action in actions.keys():
				actions[action]['blended'] = 0
				if len(actions[action]['activations']) > 0:
					actions[action]['blended'] = np.average(actions[action]['values'], weights=actions[action]['activations'])
			choice = max(actions, key=lambda action: actions[action]['blended'])
		return choice

	def learn(self, game):
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		actions = game.genI if self.player=='investor' else game.genT
		# update value of new chunks
		for t in np.arange(game.turns):
			chunk = self.learning_memory[t]
			chunk.action = actions[t]
			chunk.reward = rewards[t]
			# estimate the value of the next chunk by retrieving all similar chunks and computing their blended value
			if t==game.turns-1:
				chunk.value = chunk.reward
			else:
				next_turn = t+1
				next_coins = game.coins if self.player=="investor" else game.giveI[next_turn]*game.match
				next_value = 0
				rValues = []
				rActivations = []
				# recall all chunks in declarative memory and compare to current chunk; include them if they pass
				# activation and similarity thresholds
				for rChunk in self.declarative_memory:
					rA = rChunk.get_activation(self.episode, self.rng)
					rS = 1 if next_coins==rChunk.coins else 0
					if rA > self.thrA and rS > 0:
						rValues.append(rChunk.value)
						rActivations.append(rA)
				if len(rValues)>0:
					next_value = np.average(rValues, weights=rActivations)
				chunk.value = chunk.reward + self.gamma*next_value

		# Check if the new chunk has identical (state, action) to any previous chunk in declarative memory.
		# If so, update that chunk's triggers, rather than adding a new chunk to declarative memory
		# if not, add a new chunk to declaritive memory
		for nChunk in self.learning_memory:
			add_nChunk = True
			for rChunk in self.declarative_memory:
				if nChunk.turn==rChunk.turn and nChunk.coins==rChunk.coins and nChunk.action == rChunk.action:
					rChunk.triggers.append(nChunk.triggers[0])
					rChunk.reward = nChunk.reward
					rChunk.value = nChunk.value
					add_nChunk = False
					break
			if add_nChunk:
				self.declarative_memory.append(nChunk)
		self.episode += 1

