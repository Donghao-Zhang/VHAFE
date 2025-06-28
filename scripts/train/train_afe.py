#!/usr/bin/env python
import sys
sys.path.append('..')
sys.path.append('../..')
import wandb
import socket
import setproctitle
from pathlib import Path
import torch
# from onpolicy.config import get_config
# from onpolicy.envs.mpe.MPE_env import MPEEnv
# from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from config import get_config
from envs.afe.AFE_env import AFEEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.data_preprocess import *
import gym
gym.logger.set_level(40)
import copy
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
# mp.set_start_method('forkserver', force=True)
"""Train script for MPEs."""


def make_train_env(all_args, features, targets, actions, evaluater):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "AFE":
                if isinstance(evaluater, list):
                    env = AFEEnv(all_args, features, targets, actions, evaluater[rank])
                else:
                    env = AFEEnv(all_args, features, targets, actions, evaluater)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, features, targets, actions, evaluater):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "AFE":
                if isinstance(evaluater, list):
                    env = AFEEnv(all_args, features, targets, actions, evaluater[rank])
                else:
                    env = AFEEnv(all_args, features, targets, actions, evaluater)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    parser.add_argument('--max_iter_times', type=int, default=15)
    parser.add_argument('--max_num_nodes', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument("--greater_is_better", action='store_true',
                        default=False, help='Whether agent share the same policy')
    parser.add_argument("--no_local", action='store_true',
                        default=False, help='Whether agent share the same policy')

    parser.add_argument('--evaluatertype', type=str, default='rf', help='model')
    parser.add_argument('--dataset', type=str,
                        default='test', choices=['test', 'openml', 'secom', 'turbine',
                                                 'simulate_flat', 'simulate_random', 'uci', 'kaggle'])
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument("--remove_o1", action='store_true', default=False)
    parser.add_argument("--remove_o2", action='store_true', default=False)
    parser.add_argument("--remove_hidden", action='store_true', default=False)
    parser.add_argument("--no_train", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 生成动作
    if all_args.remove_hidden:
        actions_special = ['</s>', 'back', 'unchanged']
    else:
        actions_special = ['</s>', 'hidden', 'back', 'unchanged']
    if all_args.remove_o1:
        actions_o1 = []
    else:
        actions_o1 = ['square', 'tanh', 'round', 'log', 'sqrt', 'mmn', 'sigmoid', 'zscore', 'reciprocal']
    if all_args.remove_o2:
        actions_o2 = []
    else:
        actions_o2 = ['sum', 'minus', 'div', 'time', 'modulo']
    print("special actions are: ", actions_special)
    print("1-order actions are: ", actions_o1)
    print("2-order actions are: ", actions_o2)
    actions = actions_special + actions_o1 + actions_o2
    all_args.o2_start = len(actions_special) + len(actions_o1)
    all_args.o2_number = len(actions_o2) if len(actions_o2) > 0 else 1

    # 加载数据
    if all_args.dataset == 'test':
        features, targets, evaluater, all_args = load_test_data(all_args)
    elif all_args.dataset == 'openml':
        # data_path = "../../envs/afe/data/openml/" + all_args.dataset_name + ".arff"  # pycharm
        data_path = "../envs/afe/data/openml/" + all_args.dataset_name + ".arff"
        features, targets, evaluater, all_args = load_arff_data(data_path, all_args)
    elif all_args.dataset == 'uci':  # housing airfoil german
        # data_path = "../../envs/afe/data/uci/"  # pycharm
        data_path = "../envs/afe/data/uci/"
        features, targets, evaluater, all_args = load_uci_data(data_path, all_args.dataset_name, all_args)
    elif all_args.dataset == 'kaggle':
        # data_path = "../../envs/afe/data/kaggle/bikeshare.csv"  # pycharm
        data_path = "../envs/afe/data/kaggle/bikeshare.csv"
        features, targets, evaluater, all_args = load_kaggle_data(data_path, all_args)
    elif all_args.dataset == 'secom':
        # features, targets, evaluater, all_args = load_SECOM('../../envs/afe/data/secom', all_args)  # pycharm
        features, targets, evaluater, all_args = load_SECOM('../envs/afe/data/secom', all_args)
    elif all_args.dataset == 'turbine':
        features, targets, evaluater, all_args = load_turbine_data('../../envs/afe/data/turbine', all_args, num_dataset=1)  # pycharm
        # features, targets, evaluater, all_args = load_turbine_data('../envs/afe/data/turbine', all_args, num_dataset=1)
    elif all_args.dataset == 'simulate_flat':
        features, targets, evaluater, all_args = load_simulation_data_flat(all_args, classification=False)
    elif all_args.dataset == 'simulate_random':
        features, targets, evaluater, all_args = load_simulation_data_random(all_args, actions, use_o2=True, classification=True)
    else:
        raise ValueError
    all_args.num_agents = features.shape[1]

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
    #                0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    if all_args.dataset_name == "":
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
                  / all_args.env_name / all_args.dataset / all_args.algorithm_name / all_args.experiment_name / all_args.evaluatertype
    else:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
                  / all_args.env_name / all_args.dataset / all_args.dataset_name / all_args.algorithm_name / all_args.experiment_name / all_args.evaluatertype


    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if all_args.no_train:
            curr_run = 'run1'
            run_dir = run_dir / curr_run
        else:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
                              str(all_args.env_name) + "-" +
                              str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, features, targets, actions, evaluater)
    eval_envs = make_eval_env(all_args, features, targets, actions, evaluater) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        # from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
        from runner.shared.afe_runner import AFERunner as Runner
    else:
        # from onpolicy.runner.separated.mpe_runner import MPERunner as Runner
        from runner.separated.afe_runner import AFERunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
