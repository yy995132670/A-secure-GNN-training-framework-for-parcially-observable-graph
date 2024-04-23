import torch
import gymnasium as gym
import numpy as np
import torch_geometric
from gymnasium import spaces
import random
from greatx.training import Trainer
from gcm.gcm import DenseGCM
from gcm.edge_selectors.temporal import TemporalBackedge
from gcm.edge_selectors.dense import DenseEdge
from gcm.edge_selectors.distance import SpatialEdge
from gcm.edge_selectors.learned import LearnedEdge
from my_model import MyModel

graph_size = 128
obs_size = 2

if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'


class GNN(torch.nn.Module):
    """A simple two-layer graph neural network"""

    def __init__(self, obs_size, hidden_size=256):
        super().__init__()
        self.gc0 = torch_geometric.nn.DenseGraphConv(obs_size, hidden_size)
        self.gc1 = torch_geometric.nn.DenseGraphConv(hidden_size, hidden_size)
        self.gc2 = torch_geometric.nn.DenseGraphConv(hidden_size, hidden_size)
        self.act = torch.nn.Tanh()

    def forward(self, x, adj, weights, B, N):
        x = self.act(self.gc0(x, adj))
        x = self.act(self.gc1(x, adj))
        return self.act(self.gc2(x, adj))


class GcnEnv(gym.Env):
    metadata = {"render_modes": ["gcn"], "datasets": ["cora", "pubmed"]}

    def __init__(self, data, splits,cfg, max_b=3, save_reward=False):
        self.data = data
        self.splits = splits#分成训练集验证集合
        self.cfg = cfg

        gnn = GNN(obs_size)
        # self.gcm = DenseGCM(gnn, edge_selectors=TemporalBackedge([1]), graph_size=graph_size)
        self.gcm = DenseGCM(gnn, edge_selectors=DenseEdge(), graph_size=graph_size)
        # self.gcm = DenseGCM(gnn, edge_selectors=SpatialEdge(5, slice(0, 10)), graph_size=graph_size)
        # self.gcm = DenseGCM(gnn, edge_selectors=LearnedEdge(2), graph_size=graph_size)

        cnt = 0
        self.graph_edges = {}
        self.edges = {}  # 将edges初始化为字典
        for i in range(self.data.num_edges):
            u = self.data.edge_index[0][i]
            v = self.data.edge_index[1][i]
            edge = (u, v)  # 使用元组来存储边，以便存入集合和字典中

            self.graph_edges[i] = edge

            # 在字典中存储边的信息，键为边的索引，值为边的元组
            self.edges[i] = edge

        # self.observation_space = spaces.Discrete(cnt + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.num_edges, 256))
        self.action_space = spaces.Discrete(self.data.num_edges * 2)

        self.baseline = []
        self.max_b = 10
        self.save_reward = save_reward
        if self.save_reward:
            self.reward_matrix = np.ones((cnt + 1, self.data.num_nodes * 2))
        self.last_action = 0
        self.re_count = 0
        self.m_t = None
        self.untrusted_nodes = {}
        self.last_reward=0.1

    def _get_obs(self):
        return list(self.edges.values())

    def _get_gcm_obs(self):
        obs_tensor = torch.tensor(self._get_obs(), dtype=torch.float32)
        gcm_s, self.m_t = self.gcm(obs_tensor, self.m_t)
        # print(gcm_s)
        s_arr = gcm_s.detach().numpy()
        return s_arr

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.key = 0
        observation = self._get_gcm_obs()
        self._count = 0
        self.last_action = 0
        self.re_count = 0
        self.graph_edges = {}  # 将graph_edges初始化为集合
        self.edges = {}  # 将edges初始化为字典
        for i in range(self.data.num_edges):
            u = self.data.edge_index[0][i]
            v = self.data.edge_index[1][i]
            edge = (u, v)  # 使用元组来存储边，以便存入集合和字典中

            self.graph_edges[i] = edge
            self.edges[i] = edge

        self.m_t = None
        self.last_reward = 0.1
        self.untrusted_nodes = {}
        return observation, {}

    def update_dict(self, key):
        if key in self.untrusted_nodes:
            self.untrusted_nodes[key] += 1
        else:
            self.untrusted_nodes[key] = 1


    def step(self, action):
        if isinstance(action, np.ndarray):
            action = np.int64(action.item())
        reward_x = 0.1
        self.untrusted_nodes = {}
        truncated = False
        # if self.last_action == action:
        #     self.re_count += 1
        #     if self.re_count > 3:
        #         action = random.randint(0, (self.data.num_nodes * 2) - 1)
        #         self.re_count = 0
        #         # print(action)
        # else:
        #     self.re_count = 0
        # self.last_action = action

        self._count += 1
        print(f"action:{action}")
        #减边
        if action < self.data.num_edges and self.edges[action] in set(self.graph_edges.values()):
            self.edges[action] = ( (torch.tensor(-1, device='cuda:0')) , (torch.tensor(-1, device='cuda:0')) )

        #加边：
        elif action >= self.data.num_edges and (torch.equal(self.edges[action-self.data.num_edges][0], torch.tensor(-1, device='cuda:0')) and torch.equal(self.edges[action-self.data.num_edges][1], torch.tensor(-1, device='cuda:0'))):
            self.edges[action - self.data.num_edges] = self.graph_edges[action - self.data.num_edges]

        else:
            return (
                self._get_gcm_obs(),
                self.last_reward,
                False,
                False,
                {"acc": 0, "Untrusted nodes:": {} },
            )
        # 更新mask值
        mask = torch.BoolTensor(self.data.num_edges)
        edges_set = {tuple((u.item(), v.item())) for u, v in self.edges.values()}
        # edges_set = {(min(u.item(), v.item()), max(u.item(), v.item())) for u, v in self.edges.values()}
        for i in range(self.data.num_edges):
            u = self.data.edge_index[0][i]
            v = self.data.edge_index[1][i]
            if (min(u.item(), v.item()), max(u.item(), v.item())) in edges_set:
                mask[i] = True
            else:#被剪掉的边
                mask[i] = False
                self.update_dict(u.item())
                self.update_dict(v.item())

        print(self.untrusted_nodes)
        more_untrust_nodes = {key for key, value in self.untrusted_nodes.items() if value > 1}
        # print(mask)
        self.model = MyModel

        trainer_my_gcn = Trainer(
            self.model(self.data.x.size(-1), 12, True, True, mask),
            device=device,
            ** self.cfg,
        )
        trainer_my_gcn.fit(
            self.data, mask=(self.splits.train_nodes, self.splits.val_nodes)
        )
        logs = trainer_my_gcn.evaluate(self.data, self.splits.test_nodes)
        acc = logs["acc"]

        reward = acc - 0.4 * sum(self.baseline) / (len(self.baseline) + 0.1)
        # if action >= self.data.num_edges and reward>self.last_reward:
        #     reward=reward+0.02
        self.last_reward = reward
        # reward = acc
        self.baseline.append(acc)
        if len(self.baseline) > self.max_b:
            self.baseline = self.baseline[1:]

        observation = self._get_gcm_obs()

        if self._count > 5:
            reward_x = reward
        if len(self.untrusted_nodes) > self.data.num_nodes * 0.2:
            truncated = True

        #CiteSeer_full
        # terminated = reward_x > 0.448
        # truncated = acc > 0.74

        #PubMed
        # terminated = reward_x > 0.53
        # truncated = acc > 0.87

        #Cora
        terminated = reward_x > 0.49
        # truncated = acc > 0.82

        #Cora_ML
        # terminated = reward_x > 0.52
        # truncated = acc > 0.87

        #football
        # terminated = reward_x > 0.53
        # terminated = acc > 0.78


        print(f"POMDP_RLSTA:{self._count}\nREWARD: {reward}\tACC: {acc}\n untrusted_nodes:{more_untrust_nodes}\t")
        return (
            observation,
            reward,
            terminated,
            truncated,
            {"acc": acc, "Untrusted_nodes": more_untrust_nodes},
        )
