import pandas as pd
import numpy as np
import ast  

def read_file_reg(filename):
    df = pd.read_csv(filename)
    state = [ast.literal_eval(i) for i in df['state'].values.tolist()]
    user = df['user'].values.tolist()
    history = [np.array(ast.literal_eval(i)) for i in df['history'].values.tolist()]
    
    data = pd.DataFrame ()
    data['user'] = user
    data['state'] = state
    data['history'] = history
    return data

def read_file(filename, isTrain=True):
	df = pd.read_csv(filename)
	user = df['user'].values.tolist()
	state = [ast.literal_eval(i) for i in df['state'].values.tolist()]
	if isTrain:
		state_reward = [ast.literal_eval(i) for i in df['state_reward'].values.tolist()]
	history = [np.array(ast.literal_eval(i)) for i in df['history'].values.tolist()]
	rewards = [ast.literal_eval(i) for i in df['rewards'].values.tolist()]

	data = pd.DataFrame ()
	data['user'] = user
	data['state'] = state
	if isTrain:
		data['state_reward'] = state_reward
	data['history'] = history
	data['rewards'] = rewards
	return data

def get_orginal_data(data_name):
	train_data = read_file('../data/'+data_name+'/train_data.csv')
	test_data = read_file('../data/'+data_name+'/test_data.csv', False)

	def data_format_reverse(data, isTrain=True):
		data_set = []
		if isTrain:
			user = data['user'].to_list()
			state = data['state'].to_list()
			state_reward = data['state_reward'].to_list()
			history = data['history'].to_list()
			rewards = data['rewards'].to_list()

			for u,s,s_r,h,r in zip(user, state, state_reward, history, rewards):
				s = np.array(s)
				s_r = np.array(s_r)
				r = np.array(r)
				all_r = np.concatenate((s_r,r))
				all_item = np.concatenate((s,h))
				all_u = np.array([u] * len(all_item))
				for i,j,k in zip(all_u, all_item, all_r):
					data_set.append([i,j,k])
		else:
			user = data['user'].to_list()
			history = data['history'].to_list()
			rewards = data['rewards'].to_list()

			for u,h,r in zip(user, history, rewards):
				for j,k in zip(h,r):
					data_set.append([int(u),int(j),k])

		return data_set

	train_set = data_format_reverse(train_data, False)
	test_set = data_format_reverse(test_data, False)

	org_train_df = pd.DataFrame(train_set)
	org_train_df.to_csv('../data/'+data_name+'/train_data_org.csv', index=False, header=None)
	org_test_df = pd.DataFrame(test_set)
	org_test_df.to_csv('../data/'+data_name+'/test_data_org.csv', index=False, header=None)

	print('done!')

