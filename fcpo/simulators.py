from collections import defaultdict, namedtuple
import numpy as np
import torch

from autoassign import autoassign
from memory import Memory, Trajectory
from torch_utils.torch_utils import get_device
from gini import gini


class Simulator:
    def __init__(self, env, policy, n_trajectories, trajectory_len, **env_args):
        self.env = env
        self.policy = policy


class SinglePathSimulator(Simulator):
    def __init__(self, env, policy, n_trajectories, trajectory_len, **env_args):
        Simulator.__init__(self, env, policy, n_trajectories, trajectory_len, **env_args)
        self.item_embeddings= env_args['item_embeddings']
        self.trajectory_len = trajectory_len
        self.n_trajectories = n_trajectories
        self.nb_item = env_args['nb_item']
        self.device = env_args['device']
        self.hit_rate = []
        self.gini_coefficient = []
        self.pop_rate = []

    def run_sim(self):
        self.policy.eval()
        with torch.no_grad():
            trajectories = np.asarray([Trajectory() for i in range(self.n_trajectories)])

            ra_length = 1
#             epsilon = 0.9
            item_embeds = torch.from_numpy(self.item_embeddings).to(self.device).float()

            ave_score = 0
            ave_cost = 0
            states = self.env.reset()
#             print(states.shape)
            recommended_item_onehot = torch.FloatTensor(self.n_trajectories, self.nb_item).zero_().to(self.device)
            recommendations = []
            for t in range(self.trajectory_len): 
                policy_input = torch.FloatTensor(states).to(self.device).view(self.n_trajectories, -1)
                weight_dists = self.policy(policy_input)
                w = weight_dists.sample()
#                 print(w.shape)
                item_weights = torch.mm(w.view(-1,item_embeds.shape[1]), item_embeds.transpose(0,1)).view(self.n_trajectories, ra_length, -1)
                item_weights = torch.mul(item_weights.transpose(0,1), 1-recommended_item_onehot).reshape(states.shape[0],ra_length,-1)
                item_idxes = torch.argmax(item_weights,dim=2)

                recommendations.append(item_idxes)
                recommended_item_onehot = recommended_item_onehot.scatter_(1, item_idxes, 1)

                actions = item_embeds[item_idxes.cpu().detach()]
                states_prime, rewards, costs, info = self.env.step(actions, item_idxes)

                for i in range(len(trajectories)):
                    trajectory = trajectories[i]
                    trajectory.observations.append(policy_input[i].to(self.device).squeeze())
                    trajectory.actions.append(actions[i].to(self.device).squeeze())
                    trajectory.rewards.append(rewards[i].to(self.device).squeeze())
                    trajectory.costs.append(costs[i].to(self.device).squeeze())


                states = states_prime
                ave_score += torch.sum(info).detach().cpu()
                ave_cost += torch.sum(costs).detach().cpu()
                 
            memory = Memory(trajectories)
    
#             print(ave_score.float()/(self.trajectory_len*self.n_trajectories), ave_cost/(self.trajectory_len*self.n_trajectories))
            self.pop_rate.append(ave_cost/(self.trajectory_len*self.n_trajectories))

            recommendation_tensor = torch.cat(recommendations,1)
            idx, val = torch.unique(torch.cat(recommendations), return_counts=True)
            hr = (ave_score.float()/(self.trajectory_len*self.n_trajectories)).cpu().numpy()
            self.hit_rate.append(hr)
            
            val_ = torch.cat((val.float(),torch.zeros(self.nb_item-len(val)).to(self.device)))
            g = gini(val_.cpu().numpy())
            self.gini_coefficient.append(g)
            
            return memory