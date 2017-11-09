from collections import namedtuple, defaultdict
import numpy as np
import random
import itertools

Tile = namedtuple('Tile',['x','y'])

class World(object):
    def take_action_get_reward(self, action):
        raise NotImplementedError()

    def get_current_state(self):
        raise NotImplementedError()

    def get_curr_aval_action_set(self):
        raise NotImplementedError()

    def is_terminated(self):
        raise NotImplementedError()

    def restart(self):
        raise NotImplementedError()



class CliffWalk(World):
    def __init__(self):
        self._width = 12
        self._height = 4
        self._state_set = set()
        self._cliff_state_set = set()
        for x, y in itertools.product(range(0, self._width), range(0, self._height)):
            self._state_set.add(Tile(x=x, y=y))
        self.set_source(Tile(x=0, y=0))
        self.set_destination(Tile(x=11, y=0))
        self.restart()
        self.terminated = False
        self._action_space = set(['left','right','up','down'])

        for x in range(1,11):
            self.add_cliff(Tile(x=x, y=0))
    
    def is_terminated(self):
        return self.terminated

    def restart(self):
        self._current_state = self._source
        self.terminated = False

    def set_source(self, tile):
        self._source = tile
    
    def set_destination(self, tile):
        self._dest = tile

    def add_cliff(self, tile):
        self._cliff_state_set.add(tile)

    def is_valid_state(self, tile):
        return tile in self._state_set

    def is_cliff_state(self, tile):
        return tile in self._cliff_state_set

    def get_curr_aval_action_set(self):
        return self._action_space

    def get_current_state(self):
        return self._current_state

    def take_action_get_reward(self, action):
        if self.terminated:
            return 0
        assert(action in self._action_space)
        next_state_x = self._current_state.x
        next_state_y = self._current_state.y
        if action == 'left':
            next_state_x = next_state_x - 1
        if action == 'right':
            next_state_x = next_state_x + 1
        if action == 'down':
            next_state_y = next_state_y - 1
        if action == 'up':
            next_state_y = next_state_y + 1
        next_state = Tile(x=next_state_x, y=next_state_y)
        # if next state is cliff state, then collect -100 reward and jump back to source
        if self.is_cliff_state(next_state):
            reward = -100
            self._current_state = self._source
        # if next state is not cliff state, incur -1 reward
        else:
            reward = -1
            # if next state is dest, jump back to source
            if next_state == self._dest:
                self._current_state = self._source
                self.terminated = True
            # if next state is neither cliff nor dest, but is valid, make the transition
            elif self.is_valid_state(next_state):
                self._current_state = next_state
            # if next state is neither cliff nor dest, and is invalid, stay at current tile

        return reward

class Learner(object):
    def __init__(self, world, algorithm, greedy_epsilon=0.1, alpha=0.5, gamma=1):
        self._world = world
        self._algorithm = algorithm
        self._greedy_epsilon = greedy_epsilon
        self._alpha = alpha
        self._gamma = gamma
        self._state_action_value = defaultdict(lambda: 0)
        assert(self._algorithm in {'sarsa', 'expected-sarsa', 'q-learning'})

    def run_single_episode(self):
        self._world.restart()
        acc_reward = 0
        state = self._world.get_current_state()
        aval_actions = self._world.get_curr_aval_action_set()
        action = self.epsilon_greedy(state, aval_actions)
        while True:
            reward = self._world.take_action_get_reward(action)
            if self._world.is_terminated():
                break
            acc_reward = acc_reward + reward
            next_state = self._world.get_current_state()
            next_aval_actions = self._world.get_curr_aval_action_set()
            next_action = self.epsilon_greedy(next_state, next_aval_actions)
            if self._algorithm == 'sarsa':
                self.sarsa(state, action, reward, next_state, next_action)
            elif self._algorithm == 'expected-sarsa':
                self.expected_sarsa(state, action, reward, next_state, next_aval_actions)
            elif self._algorithm == 'q-learning':
                self.expected_sarsa(state, action, reward, next_state, next_aval_actions)
            else:
                raise TypeError('Wrong algorithm input')
            state, action = next_state, next_action
        return acc_reward

    def sarsa(self, state, action, reward, next_state, next_action):
        sarsa_target = reward + self._gamma*self._state_action_value[(next_state, next_action)]
        sarsa_error = sarsa_target - self._state_action_value[(state, action)]
        self._state_action_value[(state, action)] = self._state_action_value[(state, action)] + \
                                                    self._alpha*sarsa_error

    def expected_sarsa(self, state, action, reward, next_state, next_aval_actions):
        max_q_value_action = max((self._state_action_value[(next_state, action)], action) 
                                    for action in next_aval_actions)
        greedy_next_action = max_q_value_action[1]
        sarsa_target = reward
        sarsa_target = sarsa_target + \
                       (1 - self._greedy_epsilon) * \
                       self._gamma * self._state_action_value[(next_state, greedy_next_action)]
        for potential_next_action in next_aval_actions:
            sarsa_target = sarsa_target + \
                           self._greedy_epsilon / len(next_aval_actions) * \
                           self._gamma * self._state_action_value[(next_state, potential_next_action)]
        sarsa_error = sarsa_target - self._state_action_value[(state, action)]
        self._state_action_value[(state, action)] = self._state_action_value[(state, action)] + \
                                                    self._alpha*sarsa_error

    def q_learning(self, state, action, reward, next_state, next_aval_actions):
        max_q_value_action = max((self._state_action_value[(next_state, action)], action) 
                                    for action in next_aval_actions)
        greedy_next_action = max_q_value_action[1]
        sarsa_target = reward + \
                       self._gamma * self._state_action_value[(next_state, greedy_next_action)]
        sarsa_error = sarsa_target - self._state_action_value[(state, action)]
        self._state_action_value[(state, action)] = self._state_action_value[(state, action)] + \
                                                    self._alpha*sarsa_error

    def epsilon_greedy(self, state, aval_actions):
        greedy_flag = np.random.random() > self._greedy_epsilon
        if greedy_flag:
            max_q_value_action = max((self._state_action_value[(state, action)], action) 
                                     for action in aval_actions)
            action = max_q_value_action[1]
        else:
            action = random.sample(aval_actions, k=1)[0]
        return action


if __name__ == '__main__':
    cliff_walk = CliffWalk()
    learner = Learner(world=cliff_walk, algorithm='sarsa', greedy_epsilon=0.1, alpha=0.5, gamma=1)
    
    average_reward = 0
    for _ in range(100):
        average_reward = average_reward + learner.run_single_episode()
    print(average_reward/float(100))