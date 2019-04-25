from gym.envs.registration import register

register(
    id='kame-v0',
    entry_point='gym_kame.envs:KameEnv',
)