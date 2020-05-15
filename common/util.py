import numpy as np
import tensorflow as tf


def weighted_random_sample(weights):
    """
    :param weights: 2d tensor [n, d] containing positive weights for sampling
    :return: 1d tensor [n] with idx in [0, d) randomly sampled proportional to weights
    """
    u = tf.random_uniform(tf.shape(weights)) # Example if weights (action num) is 4 you sample one number uniformly from 0-3
    return tf.argmax(tf.log(u) / weights, axis=1)


def select_from_each_row(params, indices):
    """
    :param params: 2d tensor of shape [d1,d2]
    :param indices: 1d tensor of shape [d1] with values in [d1, d2)
    :return: 1d tensor of shape [d1] which has one value from each row of params selected with indices
    """
    sel = tf.stack([tf.range(tf.shape(params)[0]), indices], axis=1)
    return tf.gather_nd(params, sel)

# def calc_rewards(self, batch):
def calc_rewards(self, rewards, state_values, next_state_values, dones, gamma):
        '''
        Inputs
        =====================================================
        batch: tuple of state, action, reward, done, and
            next_state values from generate_episode function

        Outputs
        =====================================================
        R: np.array of discounted rewards
        G: np.array of TD-error
        '''

        # states, actions, rewards, dones, next_states = batch
        # Convert values to np.arrays
        # rewards = np.array(rewards)
        # states = np.vstack(states)
        # next_states = np.vstack(states)
        # actions = np.array(actions)
        # dones = np.array(dones)
        batch = rewards.shape[0]
        # total_steps = len(rewards)
        total_steps = rewards.size #(MINE)
        rewards = np.reshape(rewards,(rewards.size))#(rewards, (1,batch*self.n_steps))
        next_state_values = next_state_values.reshape(rewards.size)
        state_values = state_values.reshape(rewards.size)
        dones = dones.reshape(rewards.size)
        next_state_values[dones] = 0

        # R = np.zeros_like(rewards, dtype=np.float32)
        R = np.zeros_like(rewards, dtype=np.float32)
        # G = np.zeros_like(rewards, dtype=np.float32)

        for t in range(total_steps):
            last_step = min(self.n_steps, total_steps - t)

            # Look for end of episode
            check_episode_completion = dones[t:t + last_step]
            if check_episode_completion.size > 0:
                if True in check_episode_completion:
                    next_ep_completion = np.where(check_episode_completion == True)[0][0]
                    last_step = next_ep_completion

            # Sum and discount rewards
            R[t] = sum([rewards[t + n:t + n + 1] * gamma ** n for n in range(last_step)])

        if total_steps > self.n_steps:
            R[:total_steps - self.n_steps] += next_state_values[self.n_steps:]

        G = R - state_values
        return R, G # R is used as the target and G as the advantage

def calculate_n_step_reward(
        one_step_rewards: np.ndarray,
        discount: float,
        last_state_values: np.ndarray):
    """
    :param one_step_rewards: [n_env, n_timesteps]
    :param discount: scalar discount paramater
    :param last_state_values: [n_env], bootstrap from these if not done
    :return:
    """

    discount = discount ** np.arange(one_step_rewards.shape[1], -1, -1) # From shape[1] to -1 (it will stop to 0) withe step -1: px: 11,-1,-1: 11,10,...0
    reverse_rewards = np.c_[one_step_rewards, last_state_values][:, ::-1]
    full_discounted_reverse_rewards = reverse_rewards * discount
    return (np.cumsum(full_discounted_reverse_rewards, axis=1) / discount)[:, :0:-1]

def general_nstep_adv_sequential(
        mb_rewards: np.ndarray,
        mb_values: np.ndarray,
        discount: float,
        mb_done: np.ndarray,
        lambda_par: float,
        nenvs: int,
        maxsteps: int):
    gae = np.zeros((nenvs, maxsteps))
    batch = np.arange(nenvs)
    ind = np.array(np.nonzero(mb_done)) # [batch x maxsteps]. first row indicate the batch and second the index where done=1
    for b in batch:
        told = 0
        adv = []
        indices = ind[1, np.where(ind[0] == b)[0]] + 1 # take the appropriate index in which done=1 per batch b
        indices = np.insert(indices, 0, 0, axis=0) # insert a 0 in front so the first index will be 0 to ...
        if indices[-1] != (maxsteps):
            indices = np.insert(indices, indices.size, maxsteps, axis=0)
        for t in range(indices.size):
            # print('t=', t, 'told=', told)
            # print('ind_t=', indices[t], 'ind_t-1=', indices[told])
            d = mb_done[b, indices[told]:indices[t]]
            # print('done', d)
            r = mb_rewards[b, indices[told]:indices[t]]
            r = r.reshape([1, r.size])
            # print('reward', r)
            v = mb_values[b, indices[told]:indices[t] + 1]
            # print('values', v)
            batch_size, timesteps = r.shape  # [1,18]
            delta = r + discount * v[1:] * (1 - d) - v[:-1]
            delta_rev = delta[:, ::-1]
            adjustment = (discount * lambda_par) ** np.arange(timesteps, 0, -1)
            advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1]
            adv = np.concatenate([adv, advantage[0]])
            told = t
        gae[b] = adv
    return gae

def general_n_step_advantage(
        one_step_rewards: np.ndarray,
        value_estimates: np.ndarray,
        discount: float,
        dones: np.ndarray,
        lambda_par: float
):
    """
    :param one_step_rewards: [n_env, n_timesteps]
    :param value_estimates: [n_env, n_timesteps + 1]
    :param discount: "gamma" in https://arxiv.org/pdf/1707.06347.pdf and most of the rl-literature
    :param lambda_par: lambda in https://arxiv.org/pdf/1707.06347.pdf
    :return:
    """
    assert 0.0 < discount <= 1.0
    assert 0.0 <= lambda_par <= 1.0
    batch_size, timesteps = one_step_rewards.shape
    assert value_estimates.shape == (batch_size, timesteps + 1)
    delta = one_step_rewards + discount * value_estimates[:, 1:] * (1-dones) - value_estimates[:, :-1] # values: first [:,1:]=take everything from the index one and after (so the element at 0 indx will be left out
    # value-estimates[:,:-1] means take everything and leave the last element out (all operations are for each env so thats why u have ":"

    if lambda_par == 0:
        return delta

    delta_rev = delta[:, ::-1] # reverse the vector so the last value of the vec is now first
    adjustment = (discount * lambda_par) ** np.arange(timesteps, 0, -1) # np.arrange creates an array with integer values: nstep, nstep-1,..,1. E.g. timesteps=32: [32,31,30,...,1]
    advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1] # From the paper of Schulman for GAE: eq. 16
    return advantage


def combine_first_dimensions(x: np.ndarray):
    """
    :param x: array of [batch_size, time, ...]
    :returns array of [batch_size * time, ...]
    """
    first_dim = x.shape[0] * x.shape[1]
    other_dims = x.shape[2:]
    dims = (first_dim,) + other_dims
    return x.reshape(*dims)


def ravel_index_pairs(idx_pairs, n_col):
    return tf.reduce_sum(idx_pairs * np.array([n_col, 1])[np.newaxis, ...], axis=1)


def dict_of_lists_to_list_of_dicst(x: dict):
    dim = {len(v) for v in x.values()}
    assert len(dim) == 1
    dim = dim.pop()
    return [{k: x[k][i] for k in x} for i in range(dim)]
