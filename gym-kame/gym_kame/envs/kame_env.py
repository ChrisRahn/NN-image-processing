import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from itertools import chain, product
from collections import defaultdict

class KameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    FPS = 10

    def __init__(self, target=None):
        self.seed(42)
        self.viewer = None
        self.state = None
        self.tau = 0.3  # seconds b/w state updates
        
        self.grid_width = 28
        self.grid_height = 28
        self.num_pix = self.grid_width*self.grid_height
        self.pos = np.array([int(self.grid_width/2), int(self.grid_height/2)])
        self.distance_tot = 0
        
        if target == None:
            self.target = self.np_random.randint(2, size=(self.grid_width, self.grid_height))
        else:
            self.target = target
        self.grid = np.ones_like(self.target, dtype=int) # Initialize w/ blank grid
        # self.grid_render = None

        self.max_reward = (self.target.sum())**2  # TODO

        # Observation space is the position vector concat w/ grid of pixels
        self.state_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.MultiBinary(n=self.num_pix)))
        # Action space is tuple of two ints [right, down] (coord vec from top left)
        # right: the x-coord of the nib's next position, 0..grid_width
        # down: the y-coord of the nib's next position, 0..grid_height
        self.action_space = spaces.Tuple((spaces.Discrete(self.grid_width), spaces.Discrete(self.grid_height)))

        self.prev_reward = 0.0

        self.state = tuple(chain(self.pos, self.grid.flatten()))
        # assert self.state_space.contains(self.state), "The observed state was %r, should be like %s" % (self.state, state_space.sample())
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, next_pos):
        assert self.action_space.contains(next_pos), "%r (%s) invalid" % (next_pos, type(next_pos))

        delta = np.array([(n-p) for n,p in zip(next_pos, self.pos)])
        self.distance_tot += np.sqrt((abs(delta)**2).sum())

        prev_sse = (abs(self.target - self.grid).sum())
        
        while delta.any():
            next_step = np.sign(delta)
            self.pos += next_step # step one pixel in the direction of the next point
            self.grid[(self.pos[1], self.pos[0])] = 1  # set the pixel value to black
            self.grid_render[self.pos[1], self.pos[0]].set_color(0, 0, 0)  # color the pixel render to black
            delta -= next_step  # Decrement the delta

        curr_sse = (abs(self.target - self.grid).sum())**2

        done = curr_sse==0 # TODO # stopping

        reward = prev_sse - curr_sse # reward is the positive change in SSE (improvement)
        reward += -0.5  # reward penalty per step

        new_state = tuple(chain(self.pos, self.grid.flatten()))
        self.state = new_state
        self.prev_reward += reward

        return new_state, reward, done, {}
    
    def reset(self):
        res_posx, res_posy = self.np_random.randint(2, size=2)
        res_grid = self.np_random.randint(2, size=(self.grid_height, self.grid_width))
        self.grid = res_grid
        self.pos = np.array([int(self.grid_width/2), int(self.grid_height/2)])
        self.state = tuple(chain([res_posx, res_posy], res_grid.flatten()))
        return self.state
    
    def render(self, mode='human'):
        SCREEN_WIDTH = 600
        SCREEN_HEIGHT = 600
        pix_width = SCREEN_WIDTH/self.grid_width
        pix_height = SCREEN_HEIGHT/self.grid_height
        PIX_POLY = 0.5*np.array([(-pix_width,-pix_height), (-pix_width,+pix_height), (+pix_width,+pix_height), (+pix_width,-pix_height)])

        if self.viewer is None:  # Build the viewer
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)

            # Create grid render
            self.grid_render = np.empty((self.grid_width, self.grid_height), dtype=object)
            for i, j in product(range(self.grid_width), range(self.grid_height)):
                v = self.grid[i, j]
                pixel = rendering.make_polygon(v=PIX_POLY, filled=True)
                pixel.set_color(1-v, 1-v, 1-v)
                pix_trans = rendering.Transform()
                pixel.add_attr(pix_trans)
                pix_trans.set_translation(pix_width*(j+0.5), SCREEN_HEIGHT - pix_height*(i+0.5))
                self.viewer.add_geom(pixel)
                self.grid_render[i, j] = pixel

            # Create nib render
            nib = rendering.make_circle(radius=5, res=30, filled=True)
            nib.set_color(1.0, 0.0, 0.0)
            self.nib_trans = rendering.Transform()
            nib.add_attr(self.nib_trans)
            self.viewer.add_geom(nib)
            self._nib_geom = nib

        if self.state is None: return None
        
        # Update nib render
        nib_posx, nib_posy = self.state[0:2]
        self.nib_trans.set_translation(pix_width*(nib_posx+0.5), SCREEN_HEIGHT - pix_height*(nib_posy+0.5))

        # Update grid render
        # for i, j in product(range(self.grid_width), range(self.grid_height)):
        #     v = self.grid[j, i]
        #     self.grid_render[j, i].set_color(1-v, 1-v, 1-v)

        return self.viewer.render(return_rgb_array= mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
