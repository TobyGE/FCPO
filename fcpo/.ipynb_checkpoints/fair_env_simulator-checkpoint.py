import itertools
import pandas as pd
import numpy as np
import random
import csv
import time
from data_util import *
# from embeddings import *
import torch

class Environment ():
	def __init__(self, data, nb_user, nb_item, item_embeddings, user_embeddings, item_indicator, gamma, device):
		self.data = data
		self.data_length = len(data)
		self.nb_user = nb_user
		self.nb_item = nb_item
		self.item_embeddings = item_embeddings
		self.user_embeddings = user_embeddings
		self.gamma = gamma
		self.device = device
		self.item_indicator = torch.from_numpy(item_indicator).float().to(self.device)

		self.current_state = self.reset()


	def reset(self):
		self.current_user = torch.FloatTensor(self.user_embeddings[self.data['user'].to_list()]).to(self.device)
		self.current_state = self.item_embeddings[self.data['state'].to_list()]
		return self.current_state



	def step(self, actions, item_idxes):
		costs = torch.sum(self.item_indicator[item_idxes],1)
		pred = torch.squeeze(torch.matmul(actions.float().to(self.device), self.current_user.view(self.nb_user,-1,1)))
		res = (pred > 0).int().view(self.nb_user,-1)
		total_rewards = torch.sum(res * torch.FloatTensor([self.gamma**i for i in range(actions.shape[1])]).to(self.device),1)

		for i in range(res.shape[0]):
			for j in range(res.shape[1]):
				k = res[i,j]
				if k > 0:
					temp = np.append (self.current_state[i], [self.item_embeddings[item_idxes[i,j]]], axis = 0)
					self.current_state[i] = np.delete (temp, 0, axis = 0)
		
		return self.current_state, total_rewards, costs, res