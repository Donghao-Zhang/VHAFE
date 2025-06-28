from .environment import MultiAgentEnv


def AFEEnv(args, features, targets, actions, evaluater):
    # create multiagent environment
    env = MultiAgentEnv(args, features, targets, actions, evaluater)
    return env
