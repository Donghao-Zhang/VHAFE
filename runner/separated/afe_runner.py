    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tqdm import trange

# from onpolicy.utils.util import update_linear_schedule
# from onpolicy.runner.separated.base_runner import Runner
from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class AFERunner(Runner):
    def __init__(self, config):
        super(AFERunner, self).__init__(config)
        self.config = config

    def run(self):
        self.warmup()
        if self.all_args.no_train:
            improvement = self.eval(None)
            print(improvement)
            return
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        max_improvement = 0

        # self.save()

        for episode in trange(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
                    self.sim_trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, \
                sim_values, sim_actions, sim_actions_env, sim_action_log_probs, sim_masks, merge_actions_env = self.collect(step)
                    
                # Obser reward and next obs
                # temp_start = time.time()
                obs, rewards, dones, infos = self.envs.step(merge_actions_env)
                # print(time.time() - temp_start)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                       sim_values, sim_actions, sim_actions_env, sim_action_log_probs, sim_masks
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            # if episode >= self.all_args.num_mini_batch - 1:
            # episode_length, n_rollout_threads = self.rewards.shape[0:2]
            # batch_size = n_rollout_threads * episode_length
            # data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
            # mini_batch_size = data_chunks // num_mini_batch
            # 如果mini_batch_size为0会报错，即num_mini_batch不能过大
            # n_rollout_threads * episode_length // data_chunk_length > num_mini_batch
            train_infos, sim_train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {:.4f}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                total_num_steps / (end - start)))

                if self.env_name == "AFE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)
                self.log_train(sim_train_infos, total_num_steps, prefix='sim_')

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                improvement = self.eval(total_num_steps)
                if improvement > max_improvement:
                    max_improvement = improvement
                    self.save()

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

            self.sim_buffer[agent_id].share_obs[0] = share_obs.copy()
            self.sim_buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        sim_values = []
        sim_actions = []
        sim_temp_actions_env = []
        sim_action_log_probs = []
        sim_masks = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            self.sim_trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # TODO sim
            assert np.all(self.sim_buffer[agent_id].obs[step] == self.buffer[agent_id].obs[step]) and \
                   np.all(self.sim_buffer[agent_id].share_obs[step] == self.buffer[agent_id].share_obs[step])
            sim_value, sim_action, sim_action_log_prob = self.sim_trainer[agent_id].policy.get_actions(
                self.sim_buffer[agent_id].obs[step],
                self.sim_buffer[agent_id].share_obs[step].reshape(
                    self.buffer[agent_id].share_obs[step].shape[0], self.config['num_agents'], -1
                )
            )

            sim_action = _t2n(sim_action)
            # [agents, envs, dim]
            action = _t2n(action)
            values.append(_t2n(value))

            sim_action_env = np.squeeze(np.eye(self.all_args.num_agents)[sim_action], 2)
            sim_mask = np.zeros_like(sim_action)
            for rollout_id in range(action.shape[0]):
                if action[rollout_id][0] >= self.all_args.o2_start:
                    sim_mask[rollout_id][action[rollout_id][0] - self.all_args.o2_start] = 1

            sim_values.append(_t2n(sim_value))
            sim_actions.append(sim_action)
            sim_temp_actions_env.append(sim_action_env)
            sim_action_log_probs.append(_t2n(sim_action_log_prob))
            sim_masks.append(sim_mask)

            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        sim_values = np.array(sim_values).transpose(1, 0, 2, 3)
        sim_actions = np.array(sim_actions).transpose(1, 0, 2, 3)
        sim_actions_env = []
        merge_actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            one_hot_merge_action_env = []
            for a_id, temp_action_env in enumerate(sim_temp_actions_env):
                one_hot_action_env.append(temp_action_env[i])
                one_hot_merge_action_env.append((temp_actions_env[a_id][i], temp_action_env[i]))
            sim_actions_env.append(one_hot_action_env)
            merge_actions_env.append(one_hot_merge_action_env)

        sim_action_log_probs = np.array(sim_action_log_probs).transpose(1, 0, 2, 3)
        sim_masks = np.array(sim_masks).transpose(1, 0, 2, 3)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env,\
               sim_values, sim_actions, sim_actions_env, sim_action_log_probs, sim_masks, merge_actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, \
        sim_values, sim_actions, sim_actions_env, sim_action_log_probs, sim_masks = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # mask 指引修正 sim mask
        sim_masks[masks[:, :, None, :].repeat(sim_masks.shape[2], axis=2) == 0] = 0

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])
            self.sim_buffer[agent_id].insert(share_obs,
                                             np.array(list(obs[:, agent_id])),
                                             sim_actions[:, agent_id],
                                             sim_action_log_probs[:, agent_id],
                                             sim_values[:, agent_id],
                                             rewards[:, agent_id],
                                             sim_masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            eval_temp_sim_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                # TODO sim
                self.sim_trainer[agent_id].prep_rollout()
                sim_action = self.sim_trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    np.array(list(eval_obs)), deterministic=True
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                eval_temp_actions_env.append(eval_action_env)

                sim_action = sim_action.detach().cpu().numpy()
                eval_sim_actions_env = np.squeeze(np.eye(self.all_args.num_agents)[sim_action], 2)
                eval_temp_sim_actions_env.append(eval_sim_actions_env)

                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_merge_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                one_hot_merge_action_env = []
                for a_id, temp_action_env in enumerate(eval_temp_sim_actions_env):
                    one_hot_merge_action_env.append((eval_temp_actions_env[a_id][i], temp_action_env[i]))
                eval_merge_actions_env.append(one_hot_merge_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_merge_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        eval_average_reward = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            eval_average_reward.append(eval_average_episode_rewards)
            # print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))
        print("eval average episode rewards of all agents: ", str(np.mean(eval_average_reward)))

        init_pfm = self.eval_envs.get_init_pfm()[0]
        now_pfm = self.eval_envs.get_pfm()[0]
        if self.all_args.greater_is_better:
            improvement = now_pfm - init_pfm
        else:
            improvement = init_pfm - now_pfm
        print("eval initial performance: %.4f, performance after AFE: %.4f, improvement: %.4f" %
              (init_pfm, now_pfm, improvement))
        if total_num_steps is not None:
            if self.use_wandb:
                wandb.log({"eval/"+"performance": now_pfm}, step=total_num_steps)
                wandb.log({"eval/"+"initial_performance": init_pfm}, step=total_num_steps)
                wandb.log({"eval/"+"performance_improvement": improvement}, step=total_num_steps)

            self.log_train(eval_train_infos, total_num_steps)
        return improvement

    @torch.no_grad()
    def render(self):
        pass
