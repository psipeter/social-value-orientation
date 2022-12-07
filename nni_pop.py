import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nni
from statannotations.Annotator import Annotator
from matplotlib.ticker import FormatStrFormatter

from game import *
from utils import *
from plots import *
from agents import *
from T4T import *

palette = sns.color_palette("colorblind")
sns.set_palette(palette)
sns.set(context='paper', style='white', font='CMU Serif', rc={'font.size':12, 'mathtext.fontset': 'cm'})

def makePopulation(args):
	rng = np.random.RandomState(seed=args['popSeed'])
	agents = []
	for n in range(args['nAgents']):
		if args["architecture"]=="DQN":
			params = {
				"nActions": 11 if args['player']=='investor' else 31,
				"nNeurons": int(rng.normal(args['nNeurons'], 0.2*args['nNeurons'])),
				"tau": rng.normal(args['tau'], 0.2*args['tau']),
				"alpha": rng.normal(args['alpha'], 0.2*args['alpha']),
				"gamma": rng.normal(args['gamma'], 0.2*args['gamma']),
				"explore": args['explore'],
				"update": args['update'],
				"w_s": args['w_s'],
				"w_i": rng.uniform(0, args['w_i']),
				"w_o": rng.uniform(0, args['w_o']),
			}
			agent = DQN(args['player'], ID="DQN"+str(n), seed=n, nGames=15, **params)
		elif args["architecture"]=="IBL":
			agent = IBL(args['player'], ID="IBL"+str(n), seed=n, nGames=15, **params)
		elif args["architecture"]=="NEF":
			agent = NEF(args['player'], ID="NEF"+str(n), seed=n, nGames=15, **params)
		agents.append(agent)
	return agents

def selectLearners(agents, data, thr_slope=0.1, thr_p=0.1):
	IDs = [agent.ID for agent in agents]
	agentsSelected = []
	IDsSelected = []
	for i, ID in enumerate(IDs):
		D1 = data.query('ID==@ID and player=="investor"')
		res1 = scipy.stats.linregress(D1['game'], D1['coins'])
		if res1.slope>thr_slope and res1.pvalue < thr_p:
			agentsSelected.append(agents[i])
			IDsSelected.append(ID)
	print(f'{len(agentsSelected)} agents selected')
	dataSelected = data.query("ID in @IDsSelected")
	return agentsSelected, dataSelected

def addSVO(agents, data):
	dfs = []
	for agent in agents:
		ID, w_o, w_i = agent.ID, agent.w_o, agent.w_i
		D = data.query("ID == @ID").copy()
		D['w_o'] = [w_o for _ in range(D.shape[0])]
		D['w_i'] = [w_i for _ in range(D.shape[0])]
		dfs.append(D)
	labeled = pd.concat(dfs, ignore_index=True)
	return labeled

def empSimOverlap(emp, sim):
	empGen = emp['generosity'].to_numpy()
	simGen = sim['generosity'].to_numpy()
	overlap = scipy.stats.ks_2samp(empGen, simGen)[0]
	return overlap

def main(args):
	agents = makePopulation(args)
	IDs = [agent.ID for agent in agents]
	data = run(agents, nGames=args['nGames'], opponent=args["opponent"], train=True).query("ID in @IDs")

	pop, selected = selectLearners(agents, data)
	sim = addSVO(pop, selected)
	emp = pd.read_pickle("data/human_data_cleaned.pkl")
	overlap = empSimOverlap(emp, sim)
	nni.report_final_result(overlap)
	print(overlap)


if __name__ == '__main__':
	params = {
		"architecture": "DQN",
		"player": "investor",
		"opponent": "greedy",
		"nAgents": 200,
		'nGames': 15,
		"explore": 'exponential',
		"update": 'Q-learning',
		"w_s": 1,
	}

	# p2 = {
	# 	"nNeurons": 200,#,{"_type":"randint","_value":[20, 200]},
	# 	"tau": 5,#{"_type":"uniform","_value":[1.0, 20.0]},
	# 	"alpha": 0.1,#,{"_type":"loguniform","_value":[0.01, 3.0]},
	# 	"gamma": 0.1,#{"_type":"uniform","_value":[0.0, 1.0]},
	# 	"w_o": 0.4,#{"_type":"uniform","_value":[0.0, 1.0]},
	# 	"w_i": 0.2,#{"_type":"uniform","_value":[0.0, 1.0]},
	# }
	# params = params | p2

	optimized_params = nni.get_next_parameter()
	params.update(optimized_params)
	main(params)
