import numpy as np
import random
import pandas as pd
import torch
import nengo

def get_state(player, game, agent, dim=0, turn_basis=None, coin_basis=None, turn_exp=None, coin_exp=None, representation="one-hot"):
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
	if agent=="SPA":
		if representation=="ssp":
			c = game.coins if player=='investor' else game.giveI[-1]*game.match
			vector = encode_state(t, c, turn_basis, coin_basis, turn_exp, coin_exp) if t<5 else np.zeros((dim))
			return vector

def generosity(player, give, keep):
	return np.NaN if give+keep==0 and player=='trustee' else give/(give+keep)

def encode_state(t, c, turn_basis, coin_basis, turn_exp=1.0, coin_exp=1.0):
	return np.fft.ifft(turn_basis**(t*turn_exp) * coin_basis**(c*coin_exp)).real.squeeze()

def make_unitary(v):
	return v/np.absolute(v)

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