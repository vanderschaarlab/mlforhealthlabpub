import jax.numpy as np
import numpy as onp
from pkg_resources import resource_filename

def load_data(env,num_trajs=None):

    path_head = f'{env}/expert_trajs.npy'
    path = resource_filename("volume",path_head)
    data = onp.load(path,allow_pickle=True)
    data_trajs   = data.reshape(1)[0]['trajs']
    data_returns = data.reshape(1)[0]['returns']
    data_dict    = data.reshape(1)[0]

    if num_trajs is not None:
        data_trajs = data_trajs[:num_trajs]

    state_next_state = []
    action_next_action = []

    s_dim = data_trajs[0][0][0].shape[1]
    for traj in data_trajs:

        for t in range(len(traj)-1):
            s_n_s = onp.zeros((2,s_dim))
            s_n_s[0,:] = traj[t][0]
            s_n_s[1,:] = traj[t+1][0]
            state_next_state.append(s_n_s)
        
            a_n_a = onp.zeros((2,1))
            a_n_a[0] = traj[t][1]
            a_n_a[1] = traj[t+1][1]
            action_next_action.append(a_n_a)
        
    state_next_state = onp.array(state_next_state)
    state_next_state = np.array(state_next_state)

    action_next_action = onp.array(action_next_action)
    action_next_action = np.array(action_next_action)

    a_dim = (action_next_action.max() + 1).astype(np.int32)

    inputs = state_next_state
    targets = action_next_action

    return inputs,targets,a_dim,s_dim

