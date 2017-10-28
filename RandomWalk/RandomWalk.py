import numpy as np
from collections import defaultdict

LEFT_TERMINATE_STATE = -3
RIGHT_TERMINATE_STATE = 3

def get_reward(state, action):
    return  1 if state + action == RIGHT_TERMINATE_STATE else 0

def get_next_state(state, action):
    next_state = state + action
    return None if (next_state == RIGHT_TERMINATE_STATE or
                    next_state == LEFT_TERMINATE_STATE) else next_state

def get_true_state_value(state):
    return (state - LEFT_TERMINATE_STATE)/float(RIGHT_TERMINATE_STATE-LEFT_TERMINATE_STATE)

def get_episode(state, partial_episode=None):
    if partial_episode is None:
        partial_episode = []
    action = int(np.sign(np.random.random() - 0.5))
    reward = get_reward(state, action)
    next_state = get_next_state(state, action)
    partial_episode.append((state, action, reward, next_state))
    if next_state is None:
        return partial_episode
    return get_episode(next_state, partial_episode=partial_episode)

def calc_rms_error(state_value_dict):
    state_value_list = [state_value_dict[state] for state in
                        range(LEFT_TERMINATE_STATE+1, RIGHT_TERMINATE_STATE)]
    true_state_value = [get_true_state_value(state) for state in
                        range(LEFT_TERMINATE_STATE+1, RIGHT_TERMINATE_STATE)]
    error = np.array(true_state_value) - np.array(state_value_list)
    return np.sqrt(np.mean(error**2))

def time_differential_learning(num_episode, alpha, gamma=1):
    state_value_dict = defaultdict(lambda: 0.5)
    state_value_dict[None] = 0
    for _ in range(num_episode):
        for (state, _, reward, next_state) in get_episode(state=0):
            td_target = reward + gamma*state_value_dict[next_state]
            td_error = td_target - state_value_dict[state]
            state_value_dict[state] = state_value_dict[state] + alpha*td_error
    return state_value_dict

def monte_carlo_learning(num_episode, alpha, gamma=1):
    state_value_dict = defaultdict(lambda: 0.5)
    state_value_dict[None] = 0
    for _ in range(num_episode):
        episode = get_episode(state=0)
        for (state, action, reward, next_state) in episode:
            mc_target = episode[-1][-2]
            mc_erorr = mc_target - state_value_dict[state]
            state_value_dict[state] = state_value_dict[state] + alpha*mc_erorr
    return state_value_dict

def get_rms_error(num_episode, alpha, learning_algo, num_instance=100):
    ave_rms_error = 0
    for _ in range(num_instance):
        state_value_dict = learning_algo(num_episode=num_episode, alpha=alpha)
        ave_rms_error = ave_rms_error + calc_rms_error(state_value_dict)
    return ave_rms_error/float(num_instance)

#print(get_rms_error(num_episode=25, alpha=0.15, learning_algo=monte_carlo_learning,num_instance=1))

