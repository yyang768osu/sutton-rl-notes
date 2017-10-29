import numpy as np
from collections import defaultdict

LEFT_TERMINATE_STATE = -10
RIGHT_TERMINATE_STATE = 10

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

def temporal_difference_learning(num_episode, alpha, num_step_look_ahead=1):
    # Default state_value_dict to 0 to match figure 7.2
    state_value_dict = defaultdict(lambda: 0)
    for _ in range(num_episode):
        episode = get_episode(state=0)
        len_of_episode = len(episode)
        # For each step, update the TD learning
        for index, (state, _, _, _) in enumerate(episode):
            #td_target = reward
            #td_state = next_state
            #num_step_look_ahead_aval = min(len_of_episode-index, num_step_look_ahead)
            #for n in range(1, num_step_look_ahead_aval):
            #    (_, _, reward, next_state) = episode[index+n]
            #    td_target = td_target + gamma**n*reward
            #    td_state = next_state
            #td_target = (td_target +
            #             gamma**num_step_look_ahead_aval*
            #             state_value_dict[td_state])
            td_index = min(len_of_episode-1, index+num_step_look_ahead-1)
            (_, _, reward, td_state) = episode[td_index]
            td_target = reward + state_value_dict[td_state]
            td_error = td_target - state_value_dict[state]
            state_value_dict[state] = state_value_dict[state] + alpha*td_error
    return state_value_dict

def get_rms_error(num_episode, alpha, num_step_look_ahead, num_instance=100):
    ave_rms_error = 0
    for _ in range(num_instance):
        state_value_dict = temporal_difference_learning(num_episode=num_episode, 
                                                        alpha=alpha,
                                                        num_step_look_ahead=num_step_look_ahead)
        ave_rms_error = ave_rms_error + calc_rms_error(state_value_dict)
    return ave_rms_error/float(num_instance)

#print(get_rms_error(num_episode=10, alpha=0.999, num_step_look_ahead=1, num_instance=100))

