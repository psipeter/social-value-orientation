import numpy as np
import random
import pandas as pd
import torch
import nengo
import scipy

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
		if representation=="SSP":
			c = game.coins if player=='investor' else game.giveI[-1]*game.match
			return ssp_space.encode(np.array([[t, c]]))[0]

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