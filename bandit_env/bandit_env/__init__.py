from gym.envs.registration import register

register(
    id='correlatedbandit-v0',
    entry_point='bandit_env.envs:CorrelatedBanditEnv',
    max_episode_steps=100,
    nondeterministic=True,
    kwargs={'prob': 0.1}
)