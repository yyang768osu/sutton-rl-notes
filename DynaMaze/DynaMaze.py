from collections import defaultdict, namedtuple
import random
import itertools
import numpy as np

class DynaModel(object):
    def __init__(self):
        self._state_set = set()
        self._state_to_visited_action = defaultdict(set)
        self._state_action_to_reward = defaultdict(int)
        self._state_action_to_state = defaultdict(int)
        self._state_action_to_staleness = defaultdict(int)

    def simulate_model(self):
        state = random.sample(self._state_set, k=1)[0]
        action = random.sample(self._state_to_visited_action[state], k=1)[0]
        reward = self._state_action_to_reward[(state, action)]
        next_state = self._state_action_to_state[(state, action)]
        return state, action, reward, next_state

    def update_model(self, state, action, reward, next_state, aval_action_set):
        # if state is not traversed before, then add all aval_action as dummy action
        if state not in self._state_set:
            for aval_action in aval_action_set:
                self._state_to_visited_action[state].add(aval_action)
                self._state_action_to_reward[(state, aval_action)] = 0
                self._state_action_to_state[(state, aval_action)] = state
                self._state_action_to_staleness[(state, aval_action)] = -1
        # update the reward and transition
        self._state_set.add(state)
        self._state_action_to_reward[(state, action)] = reward
        self._state_action_to_state[(state, action)] = next_state
        self._state_action_to_staleness[(state, action)] = -1
        # update the staleness of all the other (state, action) pair
        for key in self._state_action_to_staleness:
            self._state_action_to_staleness[key] = self._state_action_to_staleness[key] + 1

    def get_staleness_count(self, state, action):
        return self._state_action_to_staleness[(state, action)]

class World(object):
    def take_action_get_reward(self, action):
        raise NotImplementedError()

    def get_current_state(self):
        raise NotImplementedError()

    def get_curr_aval_action_set(self):
        raise NotImplementedError()

MazeState = namedtuple('MazeState',['x','y'])

class Maze(World):
    def __init__(self, 
                 width, 
                 height,
                 source,
                 dest):
        self._width = width
        self._height = height
        self._state_set = set()
        self._blocked_state_set = set()
        for x, y in itertools.product(range(0, self._width), range(0, self._height)):
            self._state_set.add(MazeState(x=x, y=y))
        self.set_source(source)
        self.set_destination(dest)
        self._current_state = self._source
        self._action_space = set(['left','right','up','down'])

    def is_valid_maze_state(self, state):
        return (state.x >= 0 and state.x < self._width and 
                state.y >= 0 and state.y < self._height)

    def set_source(self, source):
        assert(self.is_valid_maze_state(source))
        self._source = source
    
    def set_destination(self, dest):
        assert(self.is_valid_maze_state(dest))
        self._dest = dest

    def clear_blocked_state(self):
        self._blocked_state_set = set()

    def add_blocked_state(self, blocked_state):
        assert(self.is_valid_maze_state(blocked_state))
        assert(blocked_state != self._dest)
        self._blocked_state_set.add(blocked_state)

    def remove_blocked_state(self, blocked_state):
        assert(blocked_state in self._blocked_state_set)
        self._blocked_state_set.remove(blocked_state)

    def is_blocked_state(self, state):
        return state in self._blocked_state_set

    def take_action_get_reward(self, action):
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
        next_state = MazeState(x=next_state_x, y=next_state_y)
        if next_state == self._dest:
            reward = 1
            self._current_state = self._source
        else:
            reward = 0
            if self.is_valid_maze_state(next_state) and not self.is_blocked_state(next_state):
                self._current_state = next_state
        return reward

    def get_current_state(self):
        return self._current_state

    def get_curr_aval_action_set(self):
        return self._action_space

class DynaQAlgorithm(object):
    def __init__(self, 
                 world,
                 num_sim_per_real_action=10,
                 gamma=0.9,
                 alpha=0.1,
                 greedy_epsilon=0.1,
                 staleness_bonus_epsilon=0.001):
        self._world = world
        self._num_sim_per_real_action = num_sim_per_real_action
        self._greedy_epsilon = greedy_epsilon
        self._alpha = alpha
        self._gamma = gamma
        self._staleness_bonus_epsilon = staleness_bonus_epsilon
        self.reset()

    def reset(self):
        self._dyna_model = DynaModel()
        self._state_to_aval_action = defaultdict(set)
        self._q_value = defaultdict(int)
        self._acc_reward = 0
        init_state = self._world.get_current_state()
        self._state_to_aval_action[init_state] = self._world.get_curr_aval_action_set()

    def policy(self, state):
        greedy_policy_flag = np.random.random() > self._greedy_epsilon
        if greedy_policy_flag:
            max_est_return = max([self._q_value[(state, action)]
                                  for action in self._state_to_aval_action[state]])
            action_with_max_est_return = [action
                                          for action in self._state_to_aval_action[state]
                                          if self._q_value[(state, action)] == max_est_return]
            action = random.sample(action_with_max_est_return, k=1)[0]
        else:
            action = random.sample(self._state_to_aval_action[state], k=1)[0]
        return action

    def run(self):
        # get current state
        curr_state = self._world.get_current_state()
        # invoke policy to obtain action
        action = self.policy(curr_state)
        # take action in real world and observe reward and next_state
        reward = self._world.take_action_get_reward(action)
        next_state = self._world.get_current_state()
        # accumulate reward
        self._acc_reward = self._acc_reward + reward
        # record available action in the updated state
        self._state_to_aval_action[next_state] = self._world.get_curr_aval_action_set()
        # register the transition to the dyna_model
        self._dyna_model.update_model(curr_state,
                                      action,
                                      reward,
                                      next_state,
                                      self._state_to_aval_action[curr_state])
        # Q learning
        self.q_learning(curr_state, action, reward, next_state)
        # simulate and plan
        if any([self._q_value[key]>0.0001 for key in self._q_value]):
            for _ in range(self._num_sim_per_real_action):
                # get random state
                curr_state, action, reward, next_state = self._dyna_model.simulate_model()
                self.q_learning(curr_state, action, reward, next_state)

    def q_learning(self, curr_state, action, reward, next_state):
        td_target = reward + \
                    self._staleness_bonus_epsilon*np.sqrt(self._dyna_model.get_staleness_count(curr_state, action)) + \
                    self._gamma*max(self._q_value[(next_state, next_action)]
                                    for next_action in self._state_to_aval_action[next_state])
        td_error = td_target - self._q_value[(curr_state, action)]
        self._q_value[(curr_state, action)] = self._q_value[(curr_state, action)] + self._alpha*td_error
        if self._q_value[(curr_state, action)] < 0.05:
            self._q_value[(curr_state, action)] = 0

    def reset_acc_reward(self):
        self._acc_reward = 0

    def get_acc_reward(self):
        return self._acc_reward

if __name__ == '__main__':
    maze = Maze(width=9, 
                height=6,
                source=MazeState(x=3, y=0),
                dest=MazeState(x=8, y=5))
    maze.add_blocked_state(MazeState(x=0, y=2))
    maze.add_blocked_state(MazeState(x=1, y=2))
    maze.add_blocked_state(MazeState(x=2, y=2))
    maze.add_blocked_state(MazeState(x=3, y=2))
    maze.add_blocked_state(MazeState(x=4, y=2))
    maze.add_blocked_state(MazeState(x=5, y=2))
    maze.add_blocked_state(MazeState(x=6, y=2))
    maze.add_blocked_state(MazeState(x=7, y=2))

    dyna_q_algorithm = DynaQAlgorithm(world=maze, 
                                      num_sim_per_real_action=10, 
                                      greedy_epsilon=0.1,
                                      gamma=0.9,
                                      alpha=0.5)
    for step in range(1000):
        dyna_q_algorithm.run()
        print(step, dyna_q_algorithm.get_acc_reward())
    
    maze.remove_blocked_state(MazeState(x=0, y=2))
    maze.add_blocked_state(MazeState(x=8, y=2))

    for step in range(1000, 3000):
        dyna_q_algorithm.run()
        print(step, dyna_q_algorithm.get_acc_reward())
    