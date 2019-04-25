import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class KameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    FPS = 10
    TURTLE_POLY = [
        (0, +10), (-4, -4), (+4, -4)
        ]

    def __init__(self):
        self.seed()
        self.viewer = None
        
        self.grid_width = 2
        self.grid_height = 2
        self.num_pix = self.grid_width*self.grid_height
        self.pos = (1, 1)
        self.speed = 1
        self.distance_tot = 0
        
        self.grid = np.array([[1, 1],
                              [1, 1]])

        # Observation space is the grid of pixels
        self.observation_space = spaces.MultiBinary(n=self.num_pix)
        # Action space is tuple of two ints [right, up]
        # right: the x-coord of the turtle's next position, 0..grid_width
        # up: the y-coord of the turtle's next position, 0..grid_height
        self.action_space = spaces.Tuple((spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_height)))

        self.prev_reward = None

        self.reset()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.distance_tot += np.sqrt(sum([(i-j)**2 for i,j in zip(self.pos, action)]))
        self.pos = action

        done = False # TODO
        reward = -1.0

        return self.pos, reward, done, {}

    
    def reset(self):
        pass
    
    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
    
    def close(self):
        pass