if (__name__ == "__main__"):
    import gym
    import gym_kame
    import time
    env = gym.make('kame-v0')
    env.render()
    time.sleep(100)