import numpy as np
import random
import pandas as pd
import torch
import nengo
import scipy
from nengo.dists import Uniform, Choice, UniformHypersphere
from sspspace import *

def get_state(player, game, agent, dim=0, ssp_space=None, representation="one-hot"):
	t = len(game.giveI) if player=='investor' else len(game.giveT)
	if agent=='TQ':
		index = t if player=='investor' else t * (game.coins*game.match+1) + game.investor_give[-1]*game.match
		return index
	if agent=='DQN':
		if representation=="one-hot":
			index = t if player=='investor' else t * (game.coins*game.match+1) + game.giveI[-1]*game.match
			vector = np.zeros((dim))
			vector[index] = 1
			return torch.FloatTensor(vector)
	if agent=="IBL":
		if player=="investor":
			return t, game.coins
		else:
			return t, game.giveI[-1]*game.match
	if agent=="NEF":
		if representation=="ssp":
			c = game.coins if player=='investor' else game.giveI[-1]*game.match
			return ssp_space.encode(np.array([[t, c]]))[0]
		elif representation=="onehot":
			index = t if player=='investor' else t * (game.coins*game.match+1) + game.giveI[-1]*game.match
			vector = np.zeros((dim))
			vector[index] = 1
			return vector
def generosity(player, give, keep):
	return np.NaN if give+keep==0 and player=='trustee' else give/(give+keep)

def make_unitary(v):
	return v/np.absolute(v)

def sparsity_to_x_intercept(d, p):
	sign = 1
	if p > 0.5:
		p = 1.0 - p
		sign = -1
	return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

def measure_sparsity(spikes1, spikes2):
	nNeurons = spikes1.shape[0]
	diff = []
	quiet = 0
	for n in range(nNeurons):
		if spikes1[n]+spikes2[n]>0:
			diff.append((spikes1[n]-spikes2[n]) / (spikes1[n]+spikes2[n]))
		else:
			quiet += 1
	diff = np.array(diff)
	quiet = quiet / nNeurons
	pdiff = (np.histogram(diff)[0][0] + np.histogram(diff)[0][-1]) / diff.shape[0]
	return 100*pdiff, 100*quiet

def measure_similarity(ssp1, ssp2, mode="cosine"):
    if mode=="dot":
        return np.sum(ssp1 * ssp2)
    elif mode=="cosine":
        return np.sum(ssp1 * ssp2) / (np.linalg.norm(ssp1, ord=2) * np.linalg.norm(ssp2, ord=2))

def get_rewards(player, game, w_s, w_o, w_i, normalize, gamma):
	r_s = np.array(game.rI) if player=='investor' else np.array(game.rT)
	r_o = np.array(game.rT) if player=='investor' else np.array(game.rI)
	R = w_s*r_s + w_o*r_o - w_i*np.abs(r_s-r_o)
	if normalize:
		R = R / (game.coins * game.match)
		R[:-1] = (1-gamma)*R[:-1]
	return R

class NEFEnvironment():
	def __init__(self, agent, negativeN=True):
		self.player = agent.player
		self.state = np.zeros((agent.nStates))
		self.nActions = agent.nActions
		self.rng = agent.rng
		self.update = agent.update
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
			else:
				if self.player=='investor': self.reward = rewards[-1] / (game.coins*game.match/2)
				if self.player=='trustee': self.reward = rewards[-1] / (game.coins*game.match)
	def set_explore(self, epsilon, k=2):
		self.N = np.zeros((self.nActions))
		if self.rng.uniform(0, 1) < epsilon:
			idx = self.rng.randint(self.nActions)
			print(idx)
			if self.negativeN: self.N = -k*np.ones((self.nActions))
			self.N[idx] = k
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
		past_turn_one = t > self.t1 + self.t2 + self.t3
		if phase==1: return 0  # no recall in phase 1
		if phase==2:
			if past_turn_one: return 1  # recall state s in phase 2
			else: return 0  # but only if you've taken one turn already
		if phase==3: return 0  # no recall in phase 3
	def do_reset(self, t):
		# reset WMs only for the first tR seconds of each phase
		T = t % (self.t1 + self.t2 + self.t3)
		if 0 <=T<self.tR: return 1
		elif self.t1<T<self.t1+self.tR: return 1
		elif self.t1+self.t2<T<self.t1+self.t2+self.tR: return 1
		else: return 0


def printSimilarities(agent):
	if agent.player=='investor':
		ssps = np.zeros((5, agent.nStates))
		for turn in range(5):
			ssps[turn] = agent.ssp_space.encode(np.array([[turn, 10]]))[0]
		for pair in itertools.combinations(range(5), 2):
			print(f"similarity t={pair[0]}, t={pair[1]} = {np.dot(ssps[pair[0]], ssps[pair[1]])}")
	elif agent.player=='trustee':
		ssps = np.zeros((5, 31, agent.nStates))
		for turn in range(5):
			for coin in range(31):
				ssps[turn, coin] = agent.ssp_space.encode(np.array([[turn, coin]]))[0]
		tcList = itertools.product(range(5), range(31))
		for pair in itertools.combinations(tcList, 2):
			print(f"similarity t={pair[0]}, t={pair[1]} = {np.dot(ssps[pair[0]], ssps[pair[1]])}")
	print(f"components, mean abs {np.mean(np.abs(ssps)):.3}, range {np.min(ssps)} {np.max(ssps)}")

def setEncodersIntercepts(agent, load=False, save=True, iterations=0, thrSpikeDiff=30, thrSame=0.8):

	if agent.representation=='onehot':
		agent.intercepts = Uniform(0.1, 1)
		# agent.encoders = np.eye((agent.nNeuronsState))
		agent.encoders = np.zeros((agent.nNeuronsState, agent.nStates))
		idxs = agent.rng.randint(0, agent.nStates, size=agent.nNeuronsState)
		agent.encoders[range(agent.nNeuronsState), idxs] = 1
		agent.ssp_space = None

	elif agent.representation=='ssp':
		agent.ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=agent.nStates, domain_bounds=None,
			length_scale=np.array([[agent.length_scale_turn], [agent.length_scale_coin]]))
		agent.nStates = agent.ssp_space.ssp_dim
		agent.intercepts = Choice([sparsity_to_x_intercept(agent.nStates, agent.sparsity)])
		agent.printSimilarities()
		if load:
			encoders = np.load(f"data/NEF_encoders_player{agent.player}_seed{agent.seed}.npz")['encoders']
			assert encoders.shape[0] == agent.nNeuronsState
			assert encoders.shape[1] == agent.nStates
		else:
			class NodeInput():
				def __init__(self, dim):
					self.state = np.zeros((dim))
				def set_state(self, state):
					self.state = state
				def get_state(self):
					return self.state
			sspInput = NodeInput(agent.nStates)
			encoders = agent.ssp_space.sample_grid_encoders(agent.nNeuronsState, seed=agent.seed)
			for i in range(iterations):
				print(f'iteration {i}')
				network = nengo.Network(seed=agent.seed)
				network.config[nengo.Ensemble].neuron_type = agent.neuronType
				network.config[nengo.Ensemble].max_rates = agent.maxRates
				network.config[nengo.Probe].synapse = None
				with network:
					sspNode = nengo.Node(lambda t, x: sspInput.get_state(), size_in=2, size_out=agent.nStates)
					ens = nengo.Ensemble(agent.nNeuronsState, agent.nStates, encoders=encoders, intercepts=agent.intercepts)
					nengo.Connection(sspNode, ens, synapse=None, seed=agent.seed)
					p_spikes = nengo.Probe(ens.neurons, synapse=None)
					p_decode = nengo.Probe(ens, synapse=None)
				sim = nengo.Simulator(network, progress_bar=False)
				if agent.player=='investor':
					spikes = np.zeros((5, agent.nNeuronsState))
					similarities = np.zeros((5))
					for turn in range(5):
						sim.reset(agent.seed)
						ssp = agent.ssp_space.encode(np.array([[turn, 10]]))[0]
						sspInput.set_state(ssp)
						sim.run(0.001, progress_bar=False)
						spk = sim.data[p_spikes][-1]
						spikes[turn] = spk
						similarities[turn] = np.around(np.dot(ssp, sim.data[p_decode][-1]), 2)
					print(f"similarities: {np.mean(similarities), np.min(similarities), np.max(similarities)}")
					bad_neurons = []
					for n in range(agent.nNeuronsState):
						same = 0
						for pair in itertools.combinations(range(5), 2):
							s_a = spikes[pair[0]][n]
							s_b = spikes[pair[1]][n]
							# print(n, pair[0], pair[1], s_a, s_b)
							if np.abs(s_a-s_b)<thrSpikeDiff:
								same += 1
						if same>=thrSame*5:
							bad_neurons.append(n)
				elif agent.player=='trustee':
					spikes = np.zeros((5, 31, agent.nNeuronsState))
					similarities = np.zeros((5, 31, agent.nNeuronsState))
					for turn in range(5):
						for coin in range(31):
							sim.reset(agent.seed)
							ssp = agent.ssp_space.encode(np.array([[turn, coin]]))[0]
							sspInput.set_state(ssp)
							sim.run(0.001, progress_bar=False)
							spk = sim.data[p_spikes][-1]
							spikes[turn, coin] = spk
							similarities[turn, coin] = np.around(np.dot(ssp, sim.data[p_decode][-1]), 2)
					print(f"similarities: {np.mean(similarities), np.min(similarities), np.max(similarities)}")
					bad_neurons = []
					for n in range(agent.nNeuronsState):
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
				print(f"number of bad neurons: {len(bad_neurons)} / {agent.nNeuronsState}")
				if len(bad_neurons)==0: break
				new_encoders = agent.ssp_space.sample_grid_encoders(agent.nNeuronsState, seed=agent.seed+i+1)
				for n in bad_neurons:
					encoders[n] = new_encoders[n]
			if save:
				np.savez(f"data/NEF_encoders_player{agent.player}_seed{agent.seed}.npz", encoders=encoders)
		agent.encoders = encoders
