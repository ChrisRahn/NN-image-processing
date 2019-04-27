import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import itertools

class KameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    FPS = 10

    def __init__(self):
        self.seed()
        self.viewer = None
        self.state = None
        self.tau = 0.3  # seconds b/w state updates
        
        self.grid_width = 3
        self.grid_height = 3
        self.num_pix = self.grid_width*self.grid_height
        self.pos = np.array([[1, 1]])
        self.distance_tot = 0
        
        self.grid = self.np_random.randint(2, size=(self.grid_height, self.grid_width))

        # Observation space is the grid of pixels
        self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.MultiBinary(n=self.num_pix)))
        # Action space is tuple of two ints [right, up]
        # right: the x-coord of the nib's next position, 0..grid_width
        # up: the y-coord of the nib's next position, 0..grid_height
        self.action_space = spaces.Tuple((spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_height)))

        self.prev_reward = None

        self.reset()

        self.observation = tuple(itertools.chain(self.pos, self.grid.flatten()))
        # assert self.observation_space.contains(self.observation), "The observation was %r, should be like %s" % (self.observation, observation_space.sample())
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, next_pos):
        assert self.action_space.contains(next_pos), "%r (%s) invalid" % (next_pos, type(next_pos))

        delta = np.array([(i-j) for i,j in zip(next_pos, self.pos)])
        self.distance_tot += np.sqrt(delta.sum())
        
        while any(delta):
            next_step = np.sign(delta)
            self.pos += next_step # step one pixel in the direction of the next point
            reward += self.grid[self.pos]  # increase the reward if the pixel is black
            self.grid[self.pos] = 0  # blank the pixel
            delta -= next_step  # Decrement the delta

        done = False # TODO # stopping
        reward += -0.5  # reward penalty per step

        new_state = tuple(itertools.chain(self.pos, self.grid.flatten()))
        return new_state, reward, done, {}

    
    def reset(self):
        res_posx, res_posy = self.np_random.randint(2, size=2)
        res_grid = self.np_random.randint(2, size=(self.grid_height, self.grid_width))
        self.state = tuple(itertools.chain([res_posx, res_posy], res_grid.flatten()))
        return self.state
    
    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400
        pix_width = screen_width/grid_width
        pix_height = screen_height/grid_height

        if self.viewer is None:  # Build the viewer
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            nib = rendering.make_circle(radius=10, res=30, filled=True)
            nib.set_color(1.0, 1.0, 1.0)
            self.viewer.add_geom(nib)
            self._nib_geom = nib
        
        if self.state is None: return None
        
        return self.viewer.render(return_rgb_array= mode=='rgb_array')

    def close(self):
        pass
