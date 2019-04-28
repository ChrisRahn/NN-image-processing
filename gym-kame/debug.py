if (__name__ == "__main__"):
    import gym
    import gym_kame
    import time
    env = gym.make('kame-v0')
    for i_episode in range(1):
        # observation = env.reset()
        print(env.grid[-1::-1, :])
        env.render()
        for t in range(7):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            env.render()
            print(env.grid[-1::-1, :])
            # time.sleep(2)
            if done:
                print("Episode finished after {} timesteps")
                break
    # env.close()
