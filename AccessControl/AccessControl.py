import itertools, random
from collections import namedtuple, defaultdict

QueueState = namedtuple('QueueState', ['num_aval_server', 'priority'])

class PriorityQueue(object):
    def __init__(self, num_aval_server=10, priority_level=None, p=0.06):
        self._num_aval_server = num_aval_server
        if priority_level is None:
            self._priority_level = [1, 2, 4, 8]
        else:
            self._priority_level = priority_level
        self._p = p
        self._state_set = set()
        for aval_server, priority in itertools.product(range(self._num_aval_server+1),
                                                       self._priority_level):
            self._state_set.add(QueueState(aval_server, priority))
        self._current_state = QueueState(self._num_aval_server, 1)

    def get_current_state(self):
        return self._current_state

    def get_curr_aval_action_set(self):
        if self._current_state.num_aval_server <= self._num_aval_server:
            return ['serve', 'drop']
        return ['drop', ]

    def take_action_get_reward(self, action):
        assert(action in self.get_curr_aval_action_set())
        num_aval_server_curr = self._current_state.num_aval_server
        reward = 0
        if action == 'serve':
            reward = self._current_state.priority
            num_aval_server_curr += 1
        elif action == 'drop':
            pass
        else:
            raise TypeError('Wrong action value')
        num_aval_server_next = 0
        for _ in range(num_aval_server_curr):
            if random.random() > self._p:
                num_aval_server_next += 1
        self._current_state = QueueState(num_aval_server_next, random.choice(self._priority_level))
        return reward

class Learner(object):
    def __init__(self, world, greedy_epsilon=0.1, alpha=0.01, beta=0.01):
        self._world = world
        self._greedy_epsilon = greedy_epsilon
        self._alpha = alpha
        self._beta = beta
        self._state_action_value = defaultdict(int)
        self._average_reward = 0

    def run(self, num_iter):
        state = self._world.get_current_state()
        aval_action_set = self._world.get_curr_aval_action_set()
        action = self.epsilon_greedy(state, aval_action_set)
        for _ in range(num_iter):
            reward = self._world.take_action_get_reward(action)
            next_state = self._world.get_current_state()
            next_aval_action_set = self._world.get_curr_aval_action_set()
            next_action = self.epsilon_greedy(next_state, next_aval_action_set)
            self.differential_sarsa(state, action, reward, next_state, next_action)
            state, action = next_state, next_action

    def differential_sarsa(self, state, action, reward, next_state, next_action):
        sarsa_target = reward - self._average_reward + self._state_action_value[(next_state, next_action)]
        sarsa_error = sarsa_target - self._state_action_value[(state, action)]
        self._average_reward = self._average_reward + self._beta * sarsa_error
        self._state_action_value[(state, action)] = self._state_action_value[(state, action)] + self._alpha * sarsa_error

    def epsilon_greedy(self, state, aval_actions):
        greedy_flag = random.random() > self._greedy_epsilon
        if greedy_flag:
            max_q_value_action = max((self._state_action_value[(state, action)], action) 
                                     for action in aval_actions)
            action = max_q_value_action[1]
        else:
            action = random.sample(aval_actions, k=1)[0]
        return action    

if __name__ == '__main__':
    world = PriorityQueue()
    learner = Learner(world=world)
    learner.run(num_iter=50)
