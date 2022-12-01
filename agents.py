import numpy as np
import random
import os
import torch
import scipy
import nengo
import itertools
from utils import *
from nengo.dists import Uniform, Choice, UniformHypersphere
from sspspace import *

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

	class value(torch.nn.Module):
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
			tau=1, alpha=1e-1, gamma=0.9, explore='linear', update='SARSA',
			nGames=100, w_s=1, w_o=0, w_i=0, representation="one-hot", normalize=False):
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
		self.update = update
		self.reinitialize(player)

	def reinitialize(self, player):
		self.player = player
		torch.manual_seed(self.seed)
		self.value = self.value(self.nNeurons, self.nStates, self.nActions)
		self.optimizer = torch.optim.Adam(self.value.parameters(), self.alpha)
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
		values = self.value(game_state)			
		# Choose and action based on thees values and some exploration strategy
		doExplore, randAction = self.rng.uniform(0, 1) < self.epsilon, torch.LongTensor([self.rng.randint(self.nActions)])[0]
		action = torch.argmax(values) if not doExplore else randAction
		action = action.detach().numpy()
		# translate action into environment-appropriate signal
		available = game.coins if self.player=='investor' else game.giveI[-1]*game.match  # coins available
		give, keep = action, available-action
		# update histories for learning
		self.value_history.append(values)
		self.state_history.append(game_state)
		self.action_history.append(action)
		return give, keep

	def learn(self, game):
		self.episode += 1
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		losses = []
		for t in np.arange(game.turns):
			action = self.action_history[t]
			value = self.value_history[t][action]
			reward = torch.FloatTensor([rewards[t]])
			if t==(game.turns-1):
				next_value = 0
			else:
				if self.update=='Q-learning':
					next_state = self.state_history[t+1]
					next_values = self.value(next_state)
					next_value = torch.max(next_values)
				elif self.update=='SARSA':
					next_action = self.action_history[t+1]
					next_value = self.value_history[t+1][next_action]
			delta = reward + self.gamma*next_value - value
			losses.append(delta**2)
		loss = torch.stack(losses).sum()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


class IBL():

	def __init__(self, player, ID="IBL", seed=0, nActions=11,
			decay=0.5, sigma=0.0, thrA=-2,
			tau=1, gamma=0.9, explore='linear', update='SARSA',
			nGames=100, w_s=1, w_o=0, w_i=0, representation="one-hot", normalize=False):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.normalize = normalize
		self.rng = np.random.RandomState(seed=seed)
		self.nActions = nActions
		self.nGames = nGames
		self.gamma = gamma
		self.decay = decay
		self.sigma = sigma
		self.tau = tau
		self.explore = explore
		self.representation = representation  # not used, IBL has unique representation
		self.w_s = w_s
		self.w_o = w_o
		self.w_i = w_i
		self.thrA = thrA  # activation threshold for retrieval (loading chunks from declarative into working memory)
		self.update = update  # not used, since IBL update rule is a hybrid 
		self.reinitialize(self.player)

	class Chunk():
		def __init__(self, turn, coins, action, episode, decay, sigma):
			self.turn = turn
			self.nextTurn = None
			self.coins = coins
			self.nextCoins = None
			self.action = action
			self.value = None
			self.time = episode
			self.decay = decay  # decay rate for activation
			self.sigma = sigma  # gaussian noise added to activation
			self.activation = 0.0
		def set_activation(self, episode, rng):
			A = (episode - self.time)**(-self.decay)
			self.activation = np.log(A) + rng.logistic(loc=0.0, scale=self.sigma)
			# print(episode, self.time, A, self.activation)
			return self.activation

	def reinitialize(self, player):
		self.player = player
		self.declarative_memory = []
		self.working_memory = []
		self.learning_memory = []
		self.episode = 0

	def new_game(self, game):
		self.working_memory.clear()
		self.learning_memory.clear()
		self.rng.shuffle(self.declarative_memory)
		if self.explore=='linear':
			self.epsilon = 1 - self.episode / self.nGames
		if self.explore=='exponential':
			self.epsilon = np.exp(-self.tau*self.episode / self.nGames)
		elif game.train==False:
			self.epsilon = 0

	def move(self, game):
		turn, coins = get_state(self.player, game, "IBL")
		# load chunks from declarative memory into working memory
		self.populate_working_memory(turn, coins, game)
		# select an action that immitates the best chunk in working memory
		doExplore, randAction = self.rng.uniform(0, 1) < self.epsilon, self.rng.randint(self.nActions)
		action = self.select_action() if not doExplore else randAction
		# translate action into environment-appropriate signal
		available = game.coins if self.player=='investor' else game.giveI[-1]*game.match  # coins available
		give, keep = action, available-action
		# create a new empty chunk, populate with more information in learn()
		self.learning_memory.append(self.Chunk(turn, coins, give, self.episode, self.decay, self.sigma))
		return give, keep

	def populate_working_memory(self, turn, coins, game):
		self.working_memory.clear()
		for chunk in self.declarative_memory:
			A = chunk.set_activation(self.episode, self.rng)
			S = 1 if turn==chunk.turn and coins==chunk.coins else 0
			if A > self.thrA and S > 0:
				self.working_memory.append(chunk)

	def select_action(self):
		if len(self.working_memory)==0:
			# if there are no chunks in working memory, select a random action
			action = self.rng.randint(0, self.nActions)
		else:
			# choose an action based on the activation, similarity, reward, and/or value of chunks in working memory
			values = [[] for a in range(self.nActions)]
			activations = [[] for a in range(self.nActions)]
			blended = np.zeros((self.nActions))
			for chunk in self.working_memory:
				values[chunk.action].append(chunk.value)
				activations[chunk.action].append(chunk.activation)
			for a in range(self.nActions):
				if len(activations[a]) == 0:
					blended[a] = 0
				else:
					blended[a] = np.average(values[a], weights=np.exp(activations[a]))
			action = np.argmax(blended)
		return action

	def learn(self, game):
		rewards = get_rewards(self.player, game, self.w_s, self.w_o, self.w_i, self.normalize, self.gamma)
		# update value of new chunks using retrieval and blending
		for t in np.arange(game.turns):
			chunk = self.learning_memory[t]
			if t==game.turns-1:
				chunk.value = rewards[t]
				chunk.nextTurn = None
				chunk.nextCoins = None
			else:
				nextTurn = t+1
				nextCoins = game.coins if self.player=="investor" else game.giveI[nextTurn]*game.match
				chunk.nextTurn = nextTurn
				chunk.nextCoins = nextCoins
				# load into working memory all chunks whose state is similar to the 'next state' of the current chunk
				self.working_memory.clear()
				for rChunk in self.declarative_memory:
					rA = rChunk.activation
					rS = 1 if nextTurn==rChunk.turn and nextCoins==rChunk.coins else 0
					if rA > self.thrA and rS > 0:
						self.working_memory.append(rChunk)
				# blend the value of all these chunks to estimate the value of the next state (Q(s',a'))
				if len(self.working_memory)==0:
					nextValue = 0
				else:
					mAs = [mChunk.activation for mChunk in self.working_memory]
					mQs = [mChunk.value for mChunk in self.working_memory]
					nextValue = np.average(mQs, weights=np.exp(mAs))
				# set the value of the current chunk to be the sum of immediate reward and blended value
				chunk.value = rewards[t] + self.gamma*nextValue
		# add the new chunks to declarative memory
		for nChunk in self.learning_memory:
			self.declarative_memory.append(nChunk)
		self.episode += 1



class NEF():

	def __init__(self, player, seed=0, nActions=11, ID="NEF",
			alpha=1e-7, gamma=0.6, tau=1, explore='linear', nGames=100,
			nNeuronsState=3000, nNeuronsError=3000, nNeuronsValue=300, nNeuronsChoice=300, nArrayState=200, nNeuronsMemory=300, nNeuronsIndex=300, 
			nStates=100, sparsity=0.1, length_scale=0.5, radius=1, neuronType=nengo.LIFRate(), maxRates=Uniform(300, 400), normalize=False,
			dt=1e-3, t1=2e-1, t2=2e-1, t3=2e-1, tR=2e-2,
			w_s=1, w_o=0, w_i=0):
		self.player = player
		self.ID = ID
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.nStates = nStates  # updated in reinitialize() after HexSSP generation
		self.nActions = nActions
		self.nNeuronsState = nNeuronsState
		self.nNeuronsError = nNeuronsError
		self.nNeuronsValue = nNeuronsValue
		self.nNeuronsChoice = nNeuronsChoice
		self.nArrayState = nArrayState
		self.nNeuronsMemory = nNeuronsMemory
		self.nNeuronsIndex = nNeuronsIndex
		self.dt = dt
		self.normalize = normalize
		self.gamma = gamma
		self.alpha = alpha
		self.explore = explore
		self.nGames = nGames
		self.tau = tau
		self.w_s = w_s
		self.w_o = w_o
		self.w_i = w_i
		self.t1 = t1
		self.t2 = t2
		self.t3 = t3
		self.tR = tR
		self.radius = radius
		self.neuronType = neuronType
		self.maxRates = maxRates
		self.sparsity = sparsity
		self.length_scale = length_scale
		self.reinitialize(self.player)

	def reinitialize(self, player):
		self.player = player
		self.ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=self.nStates, domain_bounds=None, length_scale=self.length_scale)
		self.nStates = self.ssp_space.ssp_dim
		self.intercepts = Choice([sparsity_to_x_intercept(self.nStates, self.sparsity)])
		self.encoders = self.generateEncoders()
		self.decoders = np.zeros((self.nNeuronsState, self.nActions))
		self.env = self.Environment(self)
		self.network = self.build_network()
		self.simulator = nengo.Simulator(self.network, dt=self.dt, seed=self.seed, progress_bar=True)
		self.episode = 0
		self.previous_state = np.zeros((self.nStates))

	def new_game(self, game):
		if self.explore=='linear':
			self.epsilon = 1 - self.episode / self.nGames
		if self.explore=='exponential':
			self.epsilon = np.exp(-self.tau*self.episode / self.nGames)
		elif game.train==False:
			self.epsilon = 0
		self.env.__init__(self)
		self.episode += 1
		self.simulator.reset(self.seed)

	def generateEncoders(self, load=False, save=True, iterations=100, thrSpikeDiff=30, thrSame=0.8):

		if load:
			encoders = np.load(f"data/NEF_encoders_player{self.player}_seed{self.seed}.npz")['encoders']
			assert encoders.shape[0] == self.nNeuronsState
			assert encoders.shape[1] == self.nStates
		else:
			class NodeInput():
				def __init__(self, dim):
					self.state = np.zeros((dim))
				def set_state(self, state):
					self.state = state
				def get_state(self):
					return self.state
			sspInput = NodeInput(self.nStates)
			encoders = self.ssp_space.sample_grid_encoders(self.nNeuronsState, seed=self.seed)
			for i in range(iterations):
				print(f'iteration {i}')
				network = nengo.Network(seed=self.seed)
				network.config[nengo.Ensemble].neuron_type = self.neuronType
				network.config[nengo.Ensemble].max_rates = self.maxRates
				network.config[nengo.Probe].synapse = None
				with network:
					sspNode = nengo.Node(lambda t, x: sspInput.get_state(), size_in=2, size_out=self.nStates)
					ens = nengo.Ensemble(self.nNeuronsState, self.nStates, encoders=encoders, intercepts=self.intercepts)
					nengo.Connection(sspNode, ens, synapse=None, seed=self.seed)
					p_spikes = nengo.Probe(ens.neurons, synapse=None)
					p_decode = nengo.Probe(ens, synapse=None)
				sim = nengo.Simulator(network, progress_bar=False)

				if self.player=='investor':
					spikes = np.zeros((5, self.nNeuronsState))
					similarities = np.zeros((5))
					for turn in range(5):
						sim.reset(self.seed)
						ssp = self.ssp_space.encode(np.array([[turn, 10]]))[0]
						sspInput.set_state(ssp)
						sim.run(0.001, progress_bar=False)
						spk = sim.data[p_spikes][-1]
						spikes[turn] = spk
						similarities[turn] = np.around(np.dot(ssp, sim.data[p_decode][-1]), 2)
					print(f"similarities: {np.mean(similarities), np.min(similarities), np.max(similarities)}")
					bad_neurons = []
					for n in range(self.nNeuronsState):
						same = 0
						for pair in itertools.combinations(range(5), 2):
							s_a = spikes[pair[0]][n]
							s_b = spikes[pair[1]][n]
							# print(n, pair[0], pair[1], s_a, s_b)
							if np.abs(s_a-s_b)<thrSpikeDiff:
								same += 1
						if same>=thrSame*5:
							bad_neurons.append(n)


				elif self.player=='trustee':
					spikes = np.zeros((5, 31, self.nNeuronsState))
					similarities = np.zeros((5, 31, self.nNeuronsState))
					for turn in range(5):
						for coin in range(31):
							sim.reset(self.seed)
							ssp = self.ssp_space.encode(np.array([[turn, coin]]))[0]
							sspInput.set_state(ssp)
							sim.run(0.001, progress_bar=False)
							spk = sim.data[p_spikes][-1]
							spikes[turn, coin] = spk
							similarities[turn, coin] = np.around(np.dot(ssp, sim.data[p_decode][-1]), 2)
					print(f"similarities: {np.mean(similarities), np.min(similarities), np.max(similarities)}")
					bad_neurons = []
					for n in range(self.nNeuronsState):
						print(n)
						same = 0
						tcList = itertools.product(range(5), range(31))
						for pair in itertools.combinations(tcList, 2):
							s_a = spikes[pair[0]][n]
							s_b = spikes[pair[1]][n]
							# print(n, pair[0], pair[1], s_a, s_b)
							if np.abs(s_a-s_b)<thrSpikeDiff:
								same += 1
						if same>=thrSame*5*31:
							bad_neurons.append(n)

				print(f"number of bad neurons: {len(bad_neurons)} / {self.nNeuronsState}")
				if len(bad_neurons)==0: break
				new_encoders = self.ssp_space.sample_grid_encoders(self.nNeuronsState, seed=self.seed+i+1)
				for n in bad_neurons:
					encoders[n] = new_encoders[n]

			if save:
				np.savez(f"data/NEF_encoders_player{self.player}_seed{self.seed}.npz", encoders=encoders)
		return encoders

	class Environment():
		def __init__(self, agent, update='SARSA', negativeN=False):
			self.player = agent.player
			self.state = np.zeros((agent.nStates))
			self.nActions = agent.nActions
			self.rng = agent.rng
			self.update = update
			self.negativeN = negativeN
			self.t1 = agent.t1
			self.t2 = agent.t2
			self.t3 = agent.t3
			self.tR = agent.tR
			self.dt = agent.dt
			self.gamma = agent.gamma
			self.w_s = agent.w_s
			self.w_o = agent.w_o
			self.w_i = agent.w_i
			self.normalize = agent.normalize
			self.reward = 0
			self.N = np.zeros((self.nActions))
		def set_state(self, state):
			self.state = state
		def set_reward(self, game):
			r_s = np.array(game.rI) if self.player=='investor' else np.array(game.rT)
			r_o = np.array(game.rT) if self.player=='investor' else np.array(game.rI)
			rewards = self.w_s*r_s + self.w_o*r_o - self.w_i*np.abs(r_s-r_o)
			if self.normalize:
				rewards = rewards / (game.coins * game.match)
				if len(rewards)==0: self.reward = 0
				elif len(rewards)<5: self.reward = (1-self.gamma)*rewards[-1]
				elif len(rewards)==5: self.reward = rewards[-1]
			else:
				if len(rewards)==0: self.reward = 0
				else: self.reward = rewards[-1] / (game.coins*game.match)
		def set_explore(self, epsilon):
			self.N = np.zeros((self.nActions))
			if self.rng.uniform(0, 1) < epsilon:
				idx = self.rng.randint(self.nActions)
				if self.negativeN: self.N = -np.ones((self.nActions))
				self.N[idx] = 1
		def get_state(self):
			return self.state
		def get_reward(self):
			return self.reward
		def get_phase(self, t):
			T = t % (self.t1 + self.t2 + self.t3)
			if 0<=T<=self.t1: return 1
			elif self.t1<T<=self.t1+self.t2: return 2
			elif self.t1+self.t2<T<=self.t1+self.t2+self.t3: return 3
			else: raise
		def get_explore(self, t):
			phase = self.get_phase(t)
			if phase==1: return self.N if self.update=='SARSA' else 0*self.N
			if phase==2: return 0*self.N  # don't explore in phase 2
			if phase==3: return self.N  # explore in phase 3
		def save_value(self, t):
			phase = self.get_phase(t)
			if phase==1: return 1  # save value Q(s',a') to WM_value in phase 1
			if phase==2: return 0  # no update of WM_value in phase 2
			if phase==3: return 0  # no update of WM_value in phase 3
		def save_state(self, t):
			phase = self.get_phase(t)
			if phase==1: return 0  # no update of WM_state in phase 1
			if phase==2: return 0  # no update of WM_state in phase 2
			if phase==3: return 1  # save state s' to WM_state in phase 3
		def save_choice(self, t):
			phase = self.get_phase(t)
			if phase==1: return 0  # no update of WM_choice in phase 1
			if phase==2: return 0  # no update of WM_choice in phase 2
			if phase==3: return 1  # save choice a' in stage 3
		def do_replay(self, t):
			phase = self.get_phase(t)
			if phase==1: return 0  # no recall in phase 1
			if phase==2: return 1  # recall state s in phase 2
			if phase==3: return 0  # no recall in phase 3
		def do_reset(self, t):
			# reset WMs only for the first tR seconds of each phase
			T = t % (self.t1 + self.t2 + self.t3)
			if 0 <=T<self.tR: return 1
			elif self.t1<T<self.t1+self.tR: return 1
			elif self.t1+self.t2<T<self.t1+self.t2+self.tR: return 1
			else: return 0


	def build_network(self):
		network = nengo.Network(seed=self.seed)
		network.config[nengo.Ensemble].neuron_type = self.neuronType
		network.config[nengo.Ensemble].max_rates = self.maxRates
		network.config[nengo.Probe].synapse = None
		with network:

			# Network Definitions
			class LearningNode(nengo.Node):
				# implements PES learning rule
				def __init__(self, nNeurons, nActions, decoders, alpha):
					self.nNeurons = nNeurons
					self.nActions = nActions
					self.size_in = nNeurons + nActions
					self.size_out = nActions
					self.decoders = decoders
					self.alpha = alpha
					super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
				def step(self, t, x):
					nNeurons = self.nNeurons
					nActions = self.nActions
					A = x[:nNeurons]  # activites from state population
					error = x[nNeurons: nNeurons+nActions]
					delta = self.alpha * A.reshape(-1, 1) * error.reshape(1, -1)
					self.decoders[:] += delta
					Q = np.dot(A, self.decoders)
					return Q

			def Gate(nNeurons, dim, seed, radius=0.3):
				# receives two inputs (e.g. states) and two gating signals (which must be opposite, e.g. [0,1] or [1,0])
				# returns input A if gateA is open, returns input B if gateB is open
				net = nengo.Network(seed=seed)
				wInh = -1e1*np.ones((nNeurons*dim, 1))
				with net:
					net.a = nengo.Node(size_in=dim)
					net.b = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.gate_a = nengo.Ensemble(nNeurons, 1)
					net.gate_b = nengo.Ensemble(nNeurons, 1)
					net.ens_a = nengo.networks.EnsembleArray(nNeurons, dim)
					net.ens_b = nengo.networks.EnsembleArray(nNeurons, dim)
					net.ens_a.add_neuron_input()
					net.ens_b.add_neuron_input()
					nengo.Connection(net.a, net.ens_a.input, synapse=None)
					nengo.Connection(net.b, net.ens_b.input, synapse=None)
					nengo.Connection(net.ens_a.output, net.output, synapse=None)
					nengo.Connection(net.ens_b.output, net.output, synapse=None)
					nengo.Connection(net.gate_a, net.ens_a.neuron_input, transform=wInh, synapse=None)
					nengo.Connection(net.gate_b, net.ens_b.neuron_input, transform=wInh, synapse=None)
				return net

			def Memory(nNeurons, dim, seed, gain=0.1, radius=1, synapse=0, onehot_cleanup=False):
				# gated difference memory, saves "state" to the memory if "gate" is open, otherwise maintains "state" in the memory
				wInh = -1e1*np.ones((nNeurons*dim, 1))
				net = nengo.Network(seed=seed)
				with net:
					net.state = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.gate = nengo.Node(size_in=1)
					net.mem = nengo.networks.EnsembleArray(nNeurons, dim, radius=radius)
					net.diff = nengo.networks.EnsembleArray(nNeurons, dim, radius=radius)
					net.diff.add_neuron_input()
					nengo.Connection(net.state, net.diff.input, synapse=None)
					nengo.Connection(net.diff.output, net.mem.input, transform=gain, synapse=synapse)
					nengo.Connection(net.mem.output, net.mem.input, synapse=synapse)
					nengo.Connection(net.mem.output, net.diff.input, transform=-1, synapse=synapse)
					nengo.Connection(net.gate, net.diff.neuron_input, transform=wInh, synapse=None)
					nengo.Connection(net.mem.output, net.output, synapse=None)
					# output for choice memory, implements cleanup of values near zero to create a one-hot vector
					if onehot_cleanup:  # ensure output is a true one-hot vector by inhibiting and rounding
						net.output_onehot = nengo.Node(size_in=dim)
						net.onehot = nengo.networks.EnsembleArray(nNeurons, dim, intercepts=Uniform(0.5,1), encoders=Choice([[1]]))
						for a in range(dim):
							nengo.Connection(net.mem.ea_ensembles[a], net.onehot.ea_ensembles[a], function=lambda x: np.around(x), synapse=None)
						nengo.Connection(net.onehot.output, net.output_onehot, synapse=None)
				return net

			def Accumulator(nNeurons, dim, seed, thr=0.9, Tff=1e-1, Tfb=-1e-1):
				# WTA selection, each dimension of "input" accumulates in a seperate integrator (one dim of an ensemble array)
				# at a rate "Tff" until one reaches a value "thr". That dimension then 'de-accumulates' each other dimension
				# at a rate "Tfb" until they reach a value of zero
				net = nengo.Network(seed=seed)
				wInh = -1e1*np.ones((nNeurons*dim, 1))
				with net:
					net.input = nengo.Node(size_in=dim)
					net.reset = nengo.Node(size_in=1)
					net.acc = nengo.networks.EnsembleArray(nNeurons, dim, intercepts=Uniform(0, 1), encoders=Choice([[1]]))
					net.inh = nengo.networks.EnsembleArray(nNeurons, dim, intercepts=Uniform(thr, 1), encoders=Choice([[1]]))
					net.output = nengo.Node(size_in=dim)
					net.acc.add_neuron_input()
					nengo.Connection(net.input, net.acc.input, synapse=None, transform=Tff)
					nengo.Connection(net.acc.output, net.acc.input, synapse=0)
					nengo.Connection(net.acc.output, net.inh.input, synapse=0)
					nengo.Connection(net.reset, net.acc.neuron_input, synapse=None, transform=wInh)
					for a in range(dim):
						for a2 in range(dim):
							if a!=a2:
								nengo.Connection(net.inh.ea_ensembles[a], net.acc.ea_ensembles[a2], synapse=0, transform=Tfb)
					nengo.Connection(net.acc.output, net.output, synapse=None)
				return net

			def Compressor(nNeurons, dim, seed):
				# receives a full vector of values and a one-hot choice vector, and takes the dot product
				# this requires inhibiting all non-chosen dimensions of "value", then summing all dimensions
				net = nengo.Network(seed=seed)
				wInh = -1e1 * np.ones((nNeurons, 1))
				with net:
					net.values = nengo.Node(size_in=dim)
					net.choice = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=1)
					net.bias = nengo.Node(np.ones((dim)))
					net.ens = nengo.networks.EnsembleArray(nNeurons, dim)
					net.inh = nengo.networks.EnsembleArray(nNeurons, dim, intercepts=Uniform(0.1,1), encoders=Choice([[1]]))
					nengo.Connection(net.values, net.ens.input, synapse=None)
					nengo.Connection(net.choice, net.inh.input, transform=-1, synapse=None)
					nengo.Connection(net.bias, net.inh.input, synapse=None)
					for a in range(dim):
						nengo.Connection(net.inh.output[a], net.ens.ea_ensembles[a].neurons, transform=wInh, synapse=None)
						nengo.Connection(net.ens.ea_ensembles[a], net.output, synapse=None)
				return net

			def Expander(nNeurons, dim, seed):
				# receives a single vector and a one-hot choice vector, and scales the one-hot vector by "value"
				# to avoid multiplication, this requires creating a new vector with each entry equal to "value", then inhibiting all but one dim
				net = nengo.Network(seed=seed)
				wInh = -1e1 * np.ones((nNeurons, 1))
				with net:
					net.value = nengo.Node(size_in=1)
					net.choice = nengo.Node(size_in=dim)
					net.output = nengo.Node(size_in=dim)
					net.bias = nengo.Node(np.ones((dim)))
					net.ens = nengo.networks.EnsembleArray(nNeurons, dim)
					net.inh = nengo.networks.EnsembleArray(nNeurons, dim, intercepts=Uniform(0.1,1), encoders=Choice([[1]]))
					nengo.Connection(net.value, net.ens.input, transform=np.ones((dim, 1)), synapse=None)
					nengo.Connection(net.choice, net.inh.input, transform=-1, synapse=None)
					nengo.Connection(net.bias, net.inh.input, synapse=None)
					nengo.Connection(net.ens.output, net.output, synapse=None)
					for a in range(dim):
						nengo.Connection(net.inh.output[a], net.ens.ea_ensembles[a].neurons, transform=wInh, synapse=None)
				return net

			# Inputs from environment and from control systems
			state_input = nengo.Node(lambda t, x: self.env.get_state(), size_in=2, size_out=self.nStates)
			reward_input = nengo.Node(lambda t, x: self.env.get_reward(), size_in=2, size_out=1)
			explore_input = nengo.Node(lambda t, x: self.env.get_explore(t), size_in=2, size_out=self.nActions)
			replay_switch = nengo.Node(lambda t, x: self.env.do_replay(t), size_in=2, size_out=1)
			save_state_switch = nengo.Node(lambda t, x: self.env.save_state(t), size_in=2, size_out=1)
			save_value_switch = nengo.Node(lambda t, x: self.env.save_value(t), size_in=2, size_out=1)
			save_choice_switch = nengo.Node(lambda t, x: self.env.save_choice(t), size_in=2, size_out=1)
			reset_switch = nengo.Node(lambda t, x: self.env.do_reset(t), size_in=2, size_out=1)

			# Nodes, Ensembles, and Networks
			state = nengo.Ensemble(self.nNeuronsState, self.nStates, encoders=self.encoders, intercepts=self.intercepts)
			critic = LearningNode(self.nNeuronsState, self.nActions, self.decoders, self.alpha)  # connection between state and value as a node
			value = nengo.networks.EnsembleArray(self.nNeuronsValue, self.nActions)
			error = nengo.Ensemble(self.nNeuronsError, 1, radius=self.radius)
			choice = Accumulator(self.nNeuronsChoice, self.nActions, self.seed+1)
			gate = Gate(self.nArrayState, self.nStates, self.seed+2, radius=self.radius)
			WM_state = Memory(self.nArrayState, self.nStates, self.seed+3, radius=self.radius)
			WM_choice = Memory(self.nNeuronsMemory, self.nActions, self.seed+4, onehot_cleanup=True)
			WM_value = Memory(self.nNeuronsMemory, 1, self.seed+5)
			compress = Compressor(self.nNeuronsIndex, self.nActions, self.seed+6)
			expand = Expander(self.nNeuronsIndex, self.nActions, self.seed+7)

			# Connections
			# phase 1-3: send the current state (stage 1 or 3) OR the recalled previous state (stage 2) to the state population
			nengo.Connection(state_input, gate.a, synapse=None)
			nengo.Connection(WM_state.output, gate.b, synapse=None)
			nengo.Connection(replay_switch, gate.gate_a, synapse=None)
			nengo.Connection(replay_switch, gate.gate_b, function=lambda x: 1-x, synapse=None)
			nengo.Connection(gate.output, state, synapse=None)
			# phase 1-3: state to value connection, computes Q function, synaptic multiply implemented with custom node "critic"
			nengo.Connection(state.neurons, critic[:self.nNeuronsState], synapse=None)
			nengo.Connection(critic, value.input, synapse=0)
			# phase 1-3: Q values sent to WTA competition in choice
			nengo.Connection(value.output, choice.input, synapse=None)
			nengo.Connection(explore_input, choice.input, synapse=None)
			nengo.Connection(reset_switch, choice.reset, synapse=None)
			# phase 1-3: compute Q(s,a) or Q(s', a') by indexing
			nengo.Connection(value.output, compress.values, synapse=None)
			nengo.Connection(choice.output, compress.choice, synapse=None)
			# phase 1: save Q(s',a')
			nengo.Connection(compress.output, WM_value.state, synapse=None)
			nengo.Connection(save_value_switch, WM_value.gate, function=lambda x: 1-x, synapse=None)
			# phase 2: compute deltaQ
			nengo.Connection(WM_value.output, error, transform=self.gamma, synapse=None)  # gamma*Q(s',a')
			nengo.Connection(reward_input, error, synapse=None)  # R(s,a)
			nengo.Connection(compress.output, error, synapse=None, transform=-1)  # -Q(s,a)
			# phase 2: expand the scalar deltaQ to a one-hot vector indexed by a, then send this to critic
			nengo.Connection(error, expand.value, synapse=None)
			nengo.Connection(WM_choice.output_onehot, expand.choice, synapse=None)
			nengo.Connection(expand.output, critic[self.nNeuronsState: self.nNeuronsState+self.nActions], synapse=None)  # [0, ..., deltaQ(s,a), ..., 0]
			# phase 2: disinhibit learning
			wInh = -1e2*np.ones((self.nNeuronsError, 1))
			nengo.Connection(replay_switch, error.neurons, function=lambda x: 1-x, transform=wInh, synapse=None)	
			# phase 3: save s' and a' to WM, overwriting previous s and a
			nengo.Connection(state_input, WM_state.state, synapse=None)
			nengo.Connection(save_state_switch, WM_state.gate, function=lambda x: 1-x, synapse=None)
			nengo.Connection(choice.output, WM_choice.state, synapse=None)
			nengo.Connection(save_choice_switch, WM_choice.gate, function=lambda x: 1-x, synapse=None)

			# Probes
			network.p_replay_switch = nengo.Probe(replay_switch)
			network.p_save_state_switch = nengo.Probe(save_state_switch)
			network.p_save_value_switch = nengo.Probe(save_value_switch)
			network.p_save_choice_switch = nengo.Probe(save_choice_switch)
			network.p_state = nengo.Probe(state)
			network.p_value = nengo.Probe(value.output)
			network.p_error = nengo.Probe(error)
			network.p_choice = nengo.Probe(choice.output)
			network.p_WM_choice = nengo.Probe(WM_choice.output)
			network.p_WM_choice_onehot = nengo.Probe(WM_choice.output_onehot)
			network.p_WM_value = nengo.Probe(WM_value.output)
			network.p_WM_state = nengo.Probe(WM_state.output)
			network.p_compress = nengo.Probe(compress.output)
			network.p_expand = nengo.Probe(expand.output)

		return network

	def cleanPrint(self, probe, t, decimals=2):
		rounded = np.around(self.simulator.data[probe], decimals)
		past = np.mean(rounded[-int(t/self.dt):], axis=0)
		return past


	def move(self, game):
		game_state = get_state(self.player, game, "NEF", dim=self.nStates, representation="SSP", ssp_space=self.ssp_space)
		self.env.set_reward(game)
		self.env.set_state(game_state)
		self.env.set_explore(self.epsilon)
		# print("reward", self.env.get_reward())
		# print('N', self.env.N)
		# print("Stage 1")
		self.simulator.run(self.t1, progress_bar=False)  # store Q(s',a*)
		print(f'current state overlap {np.dot(game_state, self.simulator.data[self.network.p_state][-1])}')
		print('value', self.cleanPrint(self.network.p_value, self.dt))
		# print('error', self.cleanPrint(self.network.p_error, self.t1))
		# print('value', np.around(self.simulator.data[self.network.p_value], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('compress', np.around(self.simulator.data[self.network.p_compress][-1], 2))		# print('reward', self.env.get_reward())
		# print('WM choice', np.around(self.simulator.data[self.network.p_WM_choice][-1], 2))
		# print('WM choice onehot', np.around(self.simulator.data[self.network.p_WM_choice_onehot][-1], 2))
		print("Stage 2")
		self.simulator.run(self.t2, progress_bar=False)  # replay Q(s,a), recall Q(s',a') from value memory, and learn
		print(f'past state overlap {np.dot(self.previous_state, self.simulator.data[self.network.p_state][-1])}')
		# print('value', self.cleanPrint(self.network.p_value, self.t2))
		# print('error', self.cleanPrint(self.network.p_error, self.t2))
		# print('expand', self.cleanPrint(self.network.p_expand, self.t1))
		# print('expand', np.around(self.simulator.data[self.network.p_expand][-3:], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('compress', np.around(self.simulator.data[self.network.p_compress][-1], 2))
		# print('WM choice', np.around(self.simulator.data[self.network.p_WM_choice][-1], 2))
		# print('WM choice onehot', np.around(self.simulator.data[self.network.p_WM_choice_onehot][-1], 2))
		# print('WM value', self.cleanPrint(self.network.p_WM_value, self.t2))
		# print('compress', self.cleanPrint(self.network.p_compress, self.t2))
		# print("Stage 3")
		self.simulator.run(self.t3, progress_bar=False)  # choose a'
		# print('error', self.cleanPrint(self.network.p_error, self.t3))
		# print('WM choice', np.around(self.simulator.data[self.network.p_WM_choice][-1], 2))
		# print('WM choice onehot', np.around(self.simulator.data[self.network.p_WM_choice_onehot][-1], 2))
		# print('value', np.around(self.simulator.data[self.network.p_value][-2], 2))
		# print('choice', np.around(self.simulator.data[self.network.p_choice][-1], 2))
		# print('compress', np.around(self.simulator.data[self.network.p_compress][-1], 2))
		# print('error', np.around(self.simulator.data[self.network.p_error][-2], 2))
		# print('expand', np.around(self.simulator.data[self.network.p_expand][-2], 2))
		choice = self.simulator.data[self.network.p_choice][-1]
		action = np.argmax(choice)
		self.previous_state = game_state
		# print('action', action, "explore", None if np.sum(self.env.N)==0 else np.argmax(self.env.N))
		# print('action', action)
		# translate action into environment-appropriate signal
		available = game.coins if self.player=='investor' else game.giveI[-1]*game.match  # coins available
		give, keep = action, available-action
		return give, keep

	def learn(self, game):
		pass