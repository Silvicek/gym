"""
The 'autonomous vehicles' environment.
"""
import gym
from gym import spaces
from gym.utils import colorize, seeding

import math
import numpy as np
from scipy.ndimage.interpolation import shift
import copy

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw
import os


import logging
logger = logging.getLogger(__name__)

width = 1000
height = 700
CT_TARGET = 3
CT_STATIC = 1
CT_CAR = 0


class Shape:
    """
    Used for handling obstacles, car and target
    """
    def __init__(self, space, r=30, x=50, y=height - 100, angle=0.5, color='orange',
                 static=True, collision_type=CT_STATIC):
        self.r = r
        if static:
            self.body = pymunk.Body(pymunk.inf, pymunk.inf)
        else:
            self.body = pymunk.Body(1, 1./2.*r**2)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, r)
        self.shape.color = THECOLORS[color]
        self.shape.elasticity = 1.0
        self.shape.angle = angle
        self.shape.collision_type = collision_type
        space.add(self.body, self.shape)


class ACar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # moved to configure - necessary for not drawing a window
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _configure(self, args=None):
        self.memory_steps = args.memory_steps
        self.action_dim = 3
        self.observation_dim = 5
        self.old_action = None
        if args.mode == 'train' or args.mode == 'test':  # Turn off visuals
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.show_sensors = False
            self.draw_screen = False
        else:
            self.show_sensors = True
            self.draw_screen = True
        # __init__()
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Turn off alpha since we don't use it.
        self.screen.set_alpha(None)

        # Global-ish.
        self.crashed = False
        self.success = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.space.add_collision_handler(CT_CAR, CT_STATIC, self._crash_handler)
        self.space.add_collision_handler(CT_CAR, CT_TARGET, self._crash_handler_target)

        # Create the car.
        self.car = Shape(self.space, r=30, x=100, y=100, color='green', static=False,
                         collision_type=CT_CAR)

        # self.dynamic = [Shape(self.space, r=30, x=200, y=200, color='orange', static=False)]
        self.dynamic = []

        # Record steps.
        self.num_steps = 0

        self.last_position = self.car.body.position

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width - 1, height), (width - 1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles
        self.obstacles = []
        self.obstacles.append(Shape(self.space, r=55, x=25, y=350, color='purple'))
        self.obstacles.append(Shape(self.space, r=95, x=250, y=550, color='purple'))
        self.target = Shape(self.space, r=10, x=600, y=60, color='orange', collision_type=CT_TARGET)

        # state = [o_{t} | a_{t-1} | o_{t-1}]
        self.state_dim = self.observation_dim + self.memory_steps * \
                                                (self.observation_dim + self.action_dim)

        self.action_space = spaces.Discrete(self.action_dim)  # forward, left, right
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.state_dim,))

        self.full_state = np.zeros(self.state_dim)

    def _step(self, action):
        self.last_position = copy.copy(self.car.body.position)
        go = 0
        left = 0
        right = 0
        if action == 0:
            go = 1
        elif action == 1:
            left = 1
        elif action == 2:
            right = -1
        # elif action == 3:
        #     go = -1

        self.car.body.angle += .2 * (left+right)

        # Move dynamic objects
        # if self.num_steps % 5 == 0:
        #     self._move_dynamic()

        driving_direction = Vec2d(1, 0).rotated(self.car.body.angle)
        self.car.body.velocity = int(100 * go) * driving_direction

        # Update the screen and stuff.
        self.space.step(1. / 10)
        self.screen.fill(THECOLORS["black"])
        draw(self.screen, self.space)
        if self.draw_screen:
            pygame.display.flip()
        self.clock.tick()

        # Get the current location and the readings there.
        x, y = self.car.body.position

        readings = self._get_sonar_readings(x, y, self.car.body.angle)
        distance = dist(self.car.body.position, self.target.body.position)/100.
        readings += [self._get_angle(), distance]
        state = np.array(readings)

        # detect end of the episode
        if self.crashed or self._out_of_bounds():
            if self.num_steps == 0:
                self.reset()
                return self.step(action)
            elif self._out_of_bounds():
                self.crashed = True

        self.num_steps += 1

        r = self.get_reward(action)

        # handle the full state shifts
        if self.memory_steps > 0:
            self.full_state = shift(self.full_state, self.action_dim+self.observation_dim)
            self.full_state[self.observation_dim:self.observation_dim+self.action_dim] = \
                bin_from_int(self.old_action, self.action_dim)
            self.full_state[:self.observation_dim] = state
            state = self.full_state
        else:
            self.full_state = state

        self.old_action = action

        return state, r, self.crashed, {}

    def _reset(self):
        self.crashed = False
        self.num_steps = 0
        self.full_state = np.zeros_like(self.full_state)

        placed = []

        for shape in self.obstacles + [self.car, self.target] + self.dynamic:
            shape.body.velocity = Vec2d(0, 0)
            shape.body.angle = np.random.random() * 2 * np.pi
            while True:  # prevent creating overlapping objects
                shape.body.position = np.random.randint(0, width), np.random.randint(0, height)
                ok = True
                for x in placed:
                    if dist(shape.body.position, x.body.position) - \
                       (shape.shape.radius + x.shape.radius) < 0:
                        ok = False
                if ok:
                    break
            placed.append(shape)

        return self._step(None)[0]

    def _crash_handler(self, space, arbiter):
        self.crashed = True
        self.success = False
        return False

    def _crash_handler_target(self, space, arbiter):
        self.crashed = True
        self.success = True
        return False

    def _get_angle(self):
        """Angle between car and the target"""
        xc, yc = self.car.body.position
        xt, yt = self.target.body.position
        angle = norm_pi(np.arctan2(yt - yc, xt - xc) - self.car.body.angle)  # [-pi,pi]
        # if abs(angle) > np.pi/2:
        #     return 10.
        return angle

    def _out_of_bounds(self):
        def oob(t, size):
            return t < 0 or t > size
        x, y = self.car.body.position
        xt, yt = self.target.body.position
        return oob(x, width) or oob(y, height) or oob(xt, width) or oob(yt, height)

    def _render(self, mode='human', close=False):
        # TODO: this
        arr = None
        if mode == 'rgb_array':
            screen = pygame.display.get_surface()
            arr = np.array(screen.get_buffer()).reshape((height, width, -1))
        return arr

    def get_reward(self, action):
        """
        Return reward for each step
        """
        r = 0
        max_distance = dist((width, height), (0., 0.))
        distance = dist(self.target.body.position, self.car.body.position)
        last_distance = dist(self.target.body.position, self.last_position)
        if self.crashed:
            if self.success:
                r = 100.
            else:
                r = -10.
        else:
            r = -0.1 + (last_distance-distance)/max_distance*10
        return r

    def _move_dynamic(self):
        for obj in self.dynamic:
            speed = np.random.randint(20, 100)
            obj.body.angle -= np.random.randint(-1, 1)
            direction = Vec2d(1, 0).rotated(obj.body.angle)
            obj.body.velocity = speed * direction

    def _get_sonar_readings(self, x, y, angle):
        readings = []
        # Make our arms.
        arm_left = self._make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self._get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self._get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self._get_arm_distance(arm_right, x, y, angle, -0.75))

        if self.show_sensors:
            pygame.display.update()

        return readings

    def _get_arm_distance(self, arm, x, y, angle, offset):
        # TODO: is there a function for this?
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self._get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = self.screen.get_at(rotated_p)
                if self._is_empty(obs) != 0:
                    return i

            if self.show_sensors:
                pygame.draw.circle(self.screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def _make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def _get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
                   (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
                   (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def _is_empty(self, reading):  # TODO: fix this stupid color-check
        if reading == THECOLORS['black'] or reading == THECOLORS['green'] \
                or reading == THECOLORS['blue'] or reading == THECOLORS['orange']:
            return 0
        else:
            return 1


def bin_from_int(a, len):
    x = np.zeros(len)
    if a is not None:
        x[a] = 1
    return x


def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def norm_pi(angle):  # norm pi [-pi, pi]
    return angle - 2*np.pi*np.floor((angle + np.pi) / (2*np.pi))

if __name__ == "__main__":
    a = np.array([1.0, 0.0, 0.0])
    env = ACar()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.monitor.close()
