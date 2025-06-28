import gym
from gym import spaces
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import math
import time
# import multiprocessing


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, args, features, targets, actions, evaluater):

        self.args = args
        self.original_features = features
        self.targets = targets
        self.actions = actions
        self.evaluater = evaluater

        self.historysize = 5
        self.binsize = 10

        self.curr_node_ids = None  # 每个agent对应当前的node的id
        self.global_node_ids = None  # 全局每个agent对应当前的node的id
        self.global_graph_states = None  # 全局graph_states，需要进行初始化
        self.iter_times = None  # 迭代的总次数
        self.num_nodes = None  # 当前graph_states中的节点数目
        self.action_count = None  # 每个动作选择的次数
        self.history_transforms = None  # 历史historysize数目的转换（动作）
        self.last_reward = None  # 上一步的reward
        self.curr_reward = None  # 当前的reward
        self.curr_feature = None  # 当前global_graph_states可以组成的特征
        self.init_pfm = None  # 初始特征集合的表现
        self.now_pfm = None  # 当前的模型表现
        self.dones = None

        # set required vectorized gym env property
        self.n = features.shape[1]

        # configure spaces
        self.obs_dim = len(self.actions)*(2+self.historysize)+4+self.binsize*2*self.targets.shape[1]*4  # 设置智能体的观测纬度
        self.action_dim = len(self.actions)  # 设置智能体的动作纬度，这里假定为一个五个纬度的
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for _ in range(self.n):
            self.action_space.append(spaces.Discrete(self.action_dim))
            self.observation_space.append(spaces.Box(-np.inf, +np.inf, shape=(self.obs_dim,), dtype=np.float32))
            share_obs_dim += self.obs_dim
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step_one_agent(self, inp):
        agent_id, action, sim_action = inp
        if action >= self.args.o2_start:
            sim_action = sim_action[action-self.args.o2_start]
        if self.dones[agent_id]:
            action = 0  # 设置action为</s>
        operator = self.actions[action]
        curr_node_id = self.curr_node_ids[agent_id]
        curr_graph_state = copy.deepcopy(self.global_graph_states[agent_id])
        curr_node_state = curr_graph_state[curr_node_id]
        feature = curr_node_state['feature']
        done = False
        if operator in {'</s>', 'hidden', 'back', 'unchanged'}:
            if operator == '</s>':
                done = True
                next_node_id = curr_node_id
            elif operator == 'hidden':
                curr_graph_state[curr_node_id]['visibility'] = False
                next_node_id = curr_node_id
            elif operator == 'unchanged':
                next_node_id = curr_node_id
            else:
                if curr_graph_state[curr_node_id]['parent'] != -1:
                    next_node_id = curr_graph_state[curr_node_id]['parent']
                else:
                    next_node_id = curr_node_id
            if self.args.no_local:
                curr_reward = 0
                performance = 0
                curr_feature = self.generate_features_local(curr_graph_state, agent_id)
            else:
                curr_feature, curr_reward, performance = self.compute_reward_and_performance(
                    curr_graph_state, agent_id, greater_is_better=self.args.greater_is_better
                )
            parent_reward = curr_node_state["reward"]
            curr_graph_state[curr_node_id]['performance'] = performance
            curr_graph_state[curr_node_id]['reward'] = curr_reward
            depth = curr_graph_state[curr_node_id]['depth']
        else:
            # 需要填充feature, performance和reward, dependence
            next_node_state = {'visibility': True, 'performance': None, 'reward': None, 'action': action, 'dependence': None,
                               'feature': None, 'depth': curr_node_state['depth'] + 1, 'parent': curr_node_id}
            dependency = None
            if operator in {'square', 'tanh', 'round'}:
                new_feature = getattr(np, operator)(feature)
            elif operator == "log":
                vmin = feature.min()
                new_feature = np.log(feature - vmin + 1) if vmin < 1 else np.log(feature)
            elif operator == "sqrt":
                vmin = feature.min()
                new_feature = np.sqrt(feature - vmin) if vmin < 0 else np.sqrt(feature)
            elif operator == "mmn":
                mmn = MinMaxScaler()
                new_feature = mmn.fit_transform(feature[:, np.newaxis]).flatten()
            elif operator == "sigmoid":
                new_feature = (1 + getattr(np, 'tanh')(feature / 2)) / 2
            elif operator == 'reciprocal':
                new_feature = 1 / feature
                new_feature[feature == 0] = 0
            elif operator == 'zscore':
                if np.var(feature) != 0:
                    new_feature = zscore(feature)
                else:
                    new_feature = None
            # order 2
            elif operator == 'sum':
                dependency = {"feature_id": sim_action, "node_id": self.global_node_ids[sim_action]}
                new_feature = feature + self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature']
            elif operator == 'minus':
                dependency = {"feature_id": sim_action, "node_id": self.global_node_ids[sim_action]}
                new_feature = feature - self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature']
            elif operator == 'div':
                dependency = {"feature_id": sim_action, "node_id": self.global_node_ids[sim_action]}
                new_feature = feature / self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature']
                new_feature[self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature'] == 0] = 0
                new_feature = np.clip(new_feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                mmn = MinMaxScaler()
                new_feature = mmn.fit_transform(new_feature[:, np.newaxis]).flatten()
            elif operator == 'time':
                dependency = {"feature_id": sim_action, "node_id": self.global_node_ids[sim_action]}
                new_feature = feature * self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature']
                new_feature = np.clip(new_feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
                mmn = MinMaxScaler()
                new_feature = mmn.fit_transform(new_feature[:, np.newaxis]).flatten()
            elif operator == 'modulo':
                dependency = {"feature_id": sim_action, "node_id": self.global_node_ids[sim_action]}
                new_feature = np.mod(feature, self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature'])
                new_feature[self.global_graph_states[sim_action][self.global_node_ids[sim_action]]['feature'] == 0] = 0
            else:
                raise ValueError

            # if not done and (self.iter_times >= self.args.max_iter_times or self.args.max_depth)
            if new_feature is None:
                new_feature = np.zeros_like(feature)
            else:
                new_feature = np.nan_to_num(new_feature)
                new_feature = np.clip(new_feature, -math.sqrt(3.4e38), math.sqrt(3.4e38))
            # 如果新生成的特征均为一个数
            # if np.all(new_feature == 0) or \
            #         (not np.any(new_feature == 0) and np.all(new_feature / new_feature.min() == 1)):
            #     next_node_state = None
            #     next_node_id = curr_node_id
            # else:
            next_node_state['feature'] = new_feature
            next_node_state['dependency'] = dependency
            next_node_id = len(curr_graph_state)
            curr_graph_state[next_node_id] = next_node_state
            if self.args.no_local:
                curr_reward = 0
                performance = 0
                curr_feature = self.generate_features_local(curr_graph_state, agent_id)
            else:
                curr_feature, curr_reward, performance = self.compute_reward_and_performance(
                    curr_graph_state, agent_id, greater_is_better=self.args.greater_is_better
                )
            # 如果新生成的特征均为一个数
            # if next_node_state is None:
            #     depth = curr_node_state["depth"]
            # else:
            curr_graph_state[next_node_id]['performance'] = performance
            curr_graph_state[next_node_id]['reward'] = curr_reward
            depth = next_node_state['depth']
            parent_reward = curr_node_state["reward"]

        curr_action = [0] * len(self.actions)
        curr_action[action] = 1
        self.action_count[agent_id][action] += 1
        self.history_transforms[agent_id] += curr_action
        self.curr_node_ids[agent_id] = next_node_id
        self.history_transforms[agent_id] = self.history_transforms[agent_id][-len(self.actions) * self.historysize:]

        state = np.concatenate(
            [curr_action, self.action_count[agent_id],
             [depth, curr_reward, self.last_reward, parent_reward], self.history_transforms[agent_id],
             self.quantile_sketch_array(curr_graph_state[next_node_id]['feature']),
             self.quantile_sketch_array(np.mean(curr_feature, axis=1)),
             self.quantile_sketch_array(np.max(curr_feature, axis=1)),
             self.quantile_sketch_array(np.min(curr_feature, axis=1))
             ], axis=0
        )

        self.num_nodes[agent_id] = len(curr_graph_state)

        # if self.iter_times >= self.args.max_iter_times or depth >= self.args.max_depth or \
        #         self.num_nodes[agent_id] >= self.args.max_num_nodes:
        if depth >= self.args.max_depth or self.num_nodes[agent_id] >= self.args.max_num_nodes:
            done = True
        self.dones[agent_id] = done
        return state, curr_reward, done, {}, curr_graph_state

    # step  this is  env.step()
    def step(self, action_n):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        输入actions纬度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        """
        actions, sim_actions = [], []
        for act in action_n:
            actions.append(act[0])
            sim_actions.append(act[1])
        action_labels = [one_label.tolist().index(1) for one_label in actions]
        sim_action_labels = []
        for one_label in sim_actions:
            sim_action_for_one_op = []
            for op in range(one_label.shape[0]):
                sim_action_for_one_op.append(one_label[op].tolist().index(1))
            sim_action_labels.append(sim_action_for_one_op)
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        new_graph_state = []

        res = map(self.step_one_agent, zip(range(self.n), action_labels, sim_action_labels))

        for state, curr_reward, done, info, curr_graph_state in res:
            sub_agent_obs.append(state)
            sub_agent_reward.append([curr_reward])
            sub_agent_done.append(done)
            sub_agent_info.append(info)
            new_graph_state.append(curr_graph_state)

        self.global_graph_states = new_graph_state
        self.global_node_ids = copy.deepcopy(self.curr_node_ids)
        self.curr_feature, self.curr_reward, self.now_pfm = self.compute_reward_and_performance(
            greater_is_better=self.args.greater_is_better
        )
        sub_agent_info = []
        for i in range(self.n):
            sub_agent_reward[i][0] = self.curr_reward + sub_agent_reward[i][0]  # global + local
            info = {'individual_reward': sub_agent_reward[i][0]}
            sub_agent_info.append(info)

        self.iter_times += 1
        return sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        self.curr_node_ids = [0] * self.n
        self.global_node_ids = [0] * self.n
        self.iter_times = 0
        self.num_nodes = [1] * self.n
        self.action_count = [[0] * self.action_dim for _ in range(self.n)]
        self.history_transforms = [[0] * (self.action_dim * self.historysize) for _ in range(self.n)]
        self.last_reward = 0
        self.curr_reward = 0
        self.dones = [False] * self.n

        # graph_states初始化
        self.global_graph_states = [{
            0: {'visibility': True, 'performance': self.init_pfm, 'reward': 0,
                'feature': self.original_features[:, i], 'depth': 0, 'parent': -1, 'action': None,
                'dependence': None}
        } for i in range(self.n)]

        self.curr_feature = self.generate_features_global()
        self.init_pfm = self.evaluater.CV2(self.curr_feature, self.targets)
        self.now_pfm = self.init_pfm

        # 生成 states
        sub_agent_obs = []
        for i in range(self.n):
            self.global_graph_states[i][0]['performance'] = self.init_pfm
            curr_action = [0] * self.action_dim
            sub_obs = np.concatenate([
                curr_action, self.action_count[i],
                [0, self.curr_reward, self.last_reward, 0],
                self.history_transforms[i], self.quantile_sketch_array(self.original_features[:, i]),
                self.quantile_sketch_array(np.mean(self.curr_feature, axis=1)),
                self.quantile_sketch_array(np.max(self.curr_feature, axis=1)),
                self.quantile_sketch_array(np.min(self.curr_feature, axis=1))
            ], axis=0)
            # sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def render(self, mode='human', close=False):
        return None

    def graph2features(self, features, graph_states):
        if graph_states is None:
            return features
        for node_id in range(len(graph_states)):
            node = graph_states[node_id]
            # 判断是否feature中已经存在

            if (node['action'] is None or self.actions[node['action']] != '</s>') and node['visibility'] \
                    and not np.any(np.all(features == node['feature'][:, None], axis=0)) and \
                    np.max(node['feature']) != np.min(node['feature']):
                features = np.concatenate([features, node['feature'][:, None]], axis=1)
        return features

    def generate_features_global(self):
        curr_features = np.empty((self.original_features.shape[0], 0))
        for graph_states in self.global_graph_states:
            curr_features = self.graph2features(curr_features, graph_states)
        return curr_features

    def generate_features_local(self, new_graph_state, agent_id):
        curr_features = np.empty((self.original_features.shape[0], 0))
        for graph_states_id, graph_states in enumerate(self.global_graph_states):
            if graph_states_id == agent_id:
                curr_features = self.graph2features(curr_features, new_graph_state)
            else:
                curr_features = self.graph2features(curr_features, graph_states)
        return curr_features

    def compute_reward_and_performance(self, new_graph_state=None, agent_id=None, greater_is_better=False):
        # print("进入compute_reward_and_performance函数")
        # s = time.time()
        if new_graph_state is None and agent_id is None:
            curr_feature = self.generate_features_global()
        else:
            curr_feature = self.generate_features_local(new_graph_state, agent_id)
        # print("生成performance前耗时为：", time.time()-s)
        performance = self.evaluater.CV2(curr_feature, self.targets)
        # print("生成performance后耗时为：", time.time()-s)
        if greater_is_better:
            # reward = performance - self.graph_states[self.curr_node_id]['performance']  # 越大越好
            reward = performance - self.now_pfm  # 越大越好
        else:
            # reward = self.graph_states[self.curr_node_id]['performance'] - performance  # 越小越好
            reward = self.now_pfm - performance  # 越小越好
        # print("退出compute_reward_and_performance函数")
        return curr_feature, reward, performance

    def compute_metric(self):
        curr_feature = self.generate_features_global()
        # print("生成performance前耗时为：", time.time()-s)
        performance = self.evaluater.CV2_test(curr_feature, self.targets)
        return performance

    # Quantile Sketch Array
    def quantile_sketch_array(self, features):
        num_targets = self.targets.shape[1]
        qsa = np.empty([0])
        for target_id in range(num_targets):
            targets = self.targets[:, target_id]
            if self.args.tasktype == "C":
                feat_0 = features[targets == 0]
                feat_1 = features[targets == 1]
            else:
                median = np.median(targets)
                feat_0 = features[targets < median]
                feat_1 = features[targets >= median]

            if len(feat_0) == 0:
                qsa0 = [0] * self.binsize
            else:
                minval, maxval = feat_0.min(), feat_0.max()
                if abs(maxval - minval) < 1e-8:
                    qsa0 = [0] * self.binsize
                else:
                    bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                    qsa0 = np.bincount(np.digitize(feat_0, bins), minlength=self.binsize).astype(float) / len(feat_0)

            if len(feat_1) == 0:
                qsa1 = [0] * self.binsize
            else:
                minval, maxval = feat_1.min(), feat_1.max()
                if abs(maxval - minval) < 1e-8:
                    qsa1 = [0] * self.binsize
                else:
                    bins = np.arange(minval, maxval, (maxval - minval) * 1.0 / self.binsize)[1:self.binsize]
                    qsa1 = np.bincount(np.digitize(feat_1, bins), minlength=self.binsize).astype(float) / len(feat_1)
            curr_qsa = np.concatenate([qsa0, qsa1])
            qsa = np.concatenate([qsa, curr_qsa])
        return qsa
