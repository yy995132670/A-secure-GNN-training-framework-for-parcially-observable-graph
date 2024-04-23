import gymnasium as gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
import torch_geometric
from env import POMDP_env
from greatx.training import Trainer
from greatx.nn.models import GCN, SGC
from my_GCN import MyGCN
import torch_geometric.utils as utils
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from greatx.utils import split_nodes
from greatx.attack.injection import AdvInjection
from greatx.nn.models import RobustGCN, MedianGCN,SimPGCN,AirGNN
from greatx.training.callbacks import ModelCheckpoint
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import degree, remove_self_loops, add_self_loops
from torch_geometric.data import Data
if torch.cuda.is_available():
    device = 'cuda'
else:
    device='cpu'


def sgraph(data,num):

    num_nodes_to_select = num #提取的子图节点大小
    total_nodes = data.num_nodes
    # 基于节点度进行重要性采样
    node_degrees = data.edge_index[0].bincount(minlength=data.num_nodes)
    # 选择度数最高的num_nodes_to_select个节点
    _, selected_nodes = torch.topk(node_degrees, num_nodes_to_select)

    edge_index_sub, edge_attr_sub = torch_geometric.utils.subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=total_nodes)

    # 创建一个新的Data对象来存储子图信息
    data = torch_geometric.data.Data(x=data.x[selected_nodes], edge_index=edge_index_sub, y=data.y[selected_nodes])
    return data

def gml_to_pyg():
    G = nx.read_gml('./community-graphs-master/gml_graphs/eurosis.gml')

    # 转换为torch_geometric数据
    data = from_networkx(G)

    # 获取图中的节点数量
    num_nodes = G.number_of_nodes()

    # 创建单位特征矩阵
    identity_features = torch.eye(num_nodes)

    # 添加节点标签
    labels = [int(node[1]['gt']) for node in G.nodes(data=True)]

    # 使用单位特征作为节点特征
    data.x = identity_features

    # 设置节点标签
    data.y = torch.tensor(labels, dtype=torch.long)

    if 'gt' in data:
        del data.gt

    # 返回包含图数据的Data对象
    return data

# root = './dataset'
# data = Planetoid(root=root, name='CiteSeer') # CiteSeer PubMed Cora
# data = CitationFull(root=root, name='PubMed')
# data = data[0]  # 获取数据集中的图数据

data = gml_to_pyg()

# data = sgraph(data, 800)

torch.save(data, './eurosis.pt')

before = []
Attack = []
Defense1 = []
Defense2 = []
Defense3 = []
Defense4 = []
Defense5 = []
Ours = []
untrust_nodes={}
untrust_num=[]
numv=1
# data_name = './CiteSeer_full.pt'
# data_name = './PubMed.pt'
# data_name = './Cora.pt'
# data_name = './Cora_ML.pt'
# data_name = './football.pt'
data_name = './eurosis.pt'
for i in range(3):
    for i in range(numv):
        data = torch.load(data_name)

        num_features = data.x.size(-1)
        num_classes = data.y.max().item() + 1
        splits = split_nodes(data.y, random_state=100, train=0.6, test=0.2, val=0.2) #train=0.6, test=0.2, val=0.2

        cfg = {
            'lr': 3e-3,  # 自定义学习率
            'weight_decay': 5e-4,  # 自定义权重衰减
            # 可以添加更多配置参数
        }

        for atk_model_name in ["AdvInjection"]:
            for gnn_model_name in ["GCN"]:#, "SGC"]:
                print(f"{atk_model_name}_{gnn_model_name}\n")
                if atk_model_name == "AdvInjection":
                    atk_model = AdvInjection
                # elif atk_model_name == "RandomInjection":
                #     atk_model = RandomInjection

                if gnn_model_name == "GCN":
                    gnn_model = GCN #MyGCN
                elif gnn_model_name == "SGC":
                    gnn_model = SGC

                # Before
                trainer_before = Trainer(gnn_model(num_features, num_classes), device=device, **cfg)
                trainer_before.fit(data, mask=(splits.train_nodes, splits.val_nodes))
                logs = trainer_before.evaluate(data, splits.test_nodes)
                acc_before = logs["acc"]
                print(f'acc_before{acc_before}')
                before.append(acc_before)

                # Attack
                attacker = atk_model(data, device=device)
                if atk_model_name == "AdvInjection":
                    attacker.setup_surrogate(trainer_before.model)
                attacker.reset()
                if atk_model_name == "AdvInjection":
                    attacker.attack(0.10, feat_budgets=10)
                else:
                    attacker.attack(0.10, feat_limits=(0, 1))
                if data_name == './football.pt':
                    attacker.data().num_nodes +=11
                logs = trainer_before.evaluate(attacker.data(), splits.test_nodes)
                acc_after = logs["acc"]
                print(f'acc_after{acc_after}')
                Attack.append(acc_after)

                # Defense compare 1：“RobustGCN”
                compare1_model = RobustGCN(num_features, num_classes)
                compare1_trainer = Trainer(compare1_model, device=device,**cfg)
                compare1_trainer.fit(
                    attacker.data(), mask=(splits.train_nodes, splits.val_nodes)
                )
                logs = compare1_trainer.evaluate(attacker.data(), splits.test_nodes)
                acc_compare1 = logs["acc"]
                print(f'RobustGCN{acc_compare1}')
                Defense1.append(acc_compare1)

                # Defense compare 2：“MedianGCN”
                compare2_model = MedianGCN(num_features, num_classes, hids=[], acts=[])
                compare2_trainer = Trainer(compare2_model, device=device,**cfg)
                compare2_trainer.fit(
                    attacker.data(), mask=(splits.train_nodes, splits.val_nodes)
                )
                logs = compare2_trainer.evaluate(attacker.data(), splits.test_nodes)
                acc_compare2 = logs["acc"]
                print(f'MedianGCN{acc_compare2}')
                Defense2.append(acc_compare2)

                # # Defense compare 3：“simp_gcn”
                # compare3_model = SimPGCN(num_features, num_classes, hids=128)
                # compare3_trainer = Trainer(compare3_model, device=device,**cfg)
                # ckp1 = ModelCheckpoint('model.pth', monitor='val_acc')
                # compare3_trainer.fit(
                #     attacker.data(), mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp1]
                # )
                # logs = compare3_trainer.evaluate(attacker.data(), splits.test_nodes)
                # acc_compare3 = logs["acc"]
                # print(f'simp_gcn{acc_compare3}')
                # Defense3.append(acc_compare3)

                # Defense compare 4：“AirGNN”
                compare4_model = AirGNN(num_features, num_classes)
                compare4_trainer = Trainer(compare4_model, device=device,**cfg)
                ckp2 = ModelCheckpoint('model.pth', monitor='val_acc')
                compare4_trainer.fit(
                    attacker.data(), mask=(splits.train_nodes, splits.val_nodes), callbacks=[ckp2]
                )
                logs = compare4_trainer.evaluate(attacker.data(), splits.test_nodes)
                acc_compare4 = logs["acc"]
                print(f'AirGNN{acc_compare4}')

                Defense4.append(acc_compare4)

                #POMDP_RLSTA Defense
                POMDP_RLSTA_env = POMDP_env.GcnEnv(data=attacker.data(), splits=splits, cfg=cfg,max_b=20)
                # POMDP_RLSTA_model = A2C('MlpPolicy', POMDP_RLSTA_env, verbose=1, learning_rate=0.001,ent_coef=0.2)
                # POMDP_RLSTA_model.learn(total_timesteps=200, log_interval=4)
                # POMDP_RLSTA_model.save(f"{gnn_model_name}_{atk_model_name}_{data_name}_POMDP_RLSTA")
                POMDP_RLSTA_model = A2C.load(f"{gnn_model_name}_{atk_model_name}_{data_name}_POMDP_RLSTA", env=POMDP_RLSTA_env)

                untrust_nodes={}
                while not untrust_nodes:
                    POMDP_RLSTA_obs, POMDP_RLSTA_info = POMDP_RLSTA_env.reset()
                    last_action = 0
                    re_count = 0
                    RLSTA_terminated=False
                    RLSTA_truncated=False
                    POMDP_RLSTA_terminated = False
                    POMDP_RLSTA_truncated = False
                    while True:
                        if not POMDP_RLSTA_terminated and not POMDP_RLSTA_truncated:
                            POMDP_RLSTA_action, POMDP_RLSTA_states = POMDP_RLSTA_model.predict(POMDP_RLSTA_obs, deterministic=False)
                            POMDP_RLSTA_obs, POMDP_RLSTA_reward, POMDP_RLSTA_terminated, POMDP_RLSTA_truncated, POMDP_RLSTA_info = POMDP_RLSTA_env.step(POMDP_RLSTA_action)

                        if POMDP_RLSTA_terminated or POMDP_RLSTA_truncated:
                            Ours.append(POMDP_RLSTA_info['acc'])
                            untrust_nodes = POMDP_RLSTA_info['Untrusted_nodes']
                            POMDP_RLSTA_obs, POMDP_RLSTA_info = POMDP_RLSTA_env.reset()
                            break

                # 计算所有节点的度数

                # 检查 edge_index 的形状
                assert attacker.data().edge_index.shape[0] == 2

                # 合并 edge_index 的两行，以考虑无向图的特性
                all_edges = torch.cat([attacker.data().edge_index[0], attacker.data().edge_index[1]], dim=0)

                node_degrees = degree(all_edges, num_nodes=attacker.data().num_nodes)

                # 设定度数阈值
                max_degree = 100
                min_degree = 1

                # 找出度数小于阈值的不可信节点
                nodes_to_remove = [node for node in untrust_nodes if ((node_degrees[node] <= max_degree) and (node_degrees[node] >= min_degree))]
                # nodes_to_remove = [node for node in range(800,880) if node_degrees[node] > degree_threshold]
                # 准备删除操作，通过创建一个掩码来排除这些节点的边
                mask = torch.ones(attacker.data().num_nodes, dtype=torch.bool)
                mask[nodes_to_remove] = False
                mask = mask.to(device)
                # 更新 edge_index，排除涉及已删除节点的边
                edge_index = attacker.data().edge_index[:, mask[attacker.data().edge_index[0]] & mask[attacker.data().edge_index[1]]]

                # 创建一个新的 Data 对象
                delet_untrust_data = Data(x=attacker.data().x.clone(), edge_index=edge_index, y=attacker.data().y.clone())
                logs = trainer_before.evaluate(delet_untrust_data, splits.test_nodes)
                delet_untrust_acc = logs["acc"]
                print(f'delet_untrust_acc {delet_untrust_acc}')

                untrust_num.append(len(nodes_to_remove))
                Defense5.append(delet_untrust_acc)


    before_acc_average=np.mean(before)
    Attack_acc_average=np.mean(Attack)
    Defense1_acc_average=np.mean(Defense1)
    Defense2_acc_average=np.mean(Defense2)
    # Defense3_acc_average=np.mean(Defense3)
    Defense4_acc_average=np.mean(Defense4)
    Ours_acc_average=np.mean(Ours)
    Defense5_acc_average=np.mean(Defense5)
    untrust_num_average=np.mean(untrust_num)



    with open("./result.txt", "a") as file:
        file.write(f"AdvInjection_GCN{data_name}\n")
        file.write(
            f"CLN: {before_acc_average:.3};\t After AdvInjection_GCN ATK: {Attack_acc_average:.3};\t"
            f"POMDP_RLSTA: {Ours_acc_average:.3};\t"
            f"RobustGCN:{Defense1_acc_average:.3};\t"
            f"MedianGCN:{Defense2_acc_average:.3};\t"
            # f"simp_gcn:{Defense3_acc_average:.3};\t"
            f"AirGNN:{Defense4_acc_average:.3}\t"
            f"Delet_Untrust {Defense5_acc_average:.3}\n"
            f"Untrusted_nodes_num: {untrust_num_average}\t"
            f"Untrusted_nodes: {untrust_nodes}\n\n"
        )
    print(
        f"CLN: {before_acc_average:.3};\t After AdvInjection_GCN ATK: {Attack_acc_average:.3};\t"
        f"POMDP_RLSTA: {Ours_acc_average:.3};\t"
        f"RobustGCN:{Defense1_acc_average:.3};\t"
        f"MedianGCN:{Defense2_acc_average:.3};\t"
        # f"simp_gcn:{Defense3_acc_average:.3};\t"
        f"AirGNN:{Defense4_acc_average:.3}\t"
        f"Delet_Untrust {Defense5_acc_average:.3}\n"
        f"Untrusted_nodes_num: {untrust_num_average}\t"
        f"Untrusted_nodes: {untrust_nodes}\n\n"
    )
