if (__name__ == "__main__"):
    import gym
    import gym_kame
    import time
    env = gym.make('kame-v0')
    for i_episode in range(5):
        print("---BEGIN NEW EPISODE---")
        state = env.reset()
        print(env.grid)
        env.render()
        for t in range(50):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            env.render()
            print(env.grid)
            print(f"Pos: {state[0:2]}")
            # time.sleep(0.1)
            if done:
                print("Episode finished after {} timesteps")
                break
    # env.close()
