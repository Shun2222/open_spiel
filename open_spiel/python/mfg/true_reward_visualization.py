from games.predator_prey import *

def create_rew_input(obs_shape, nacs, horizon, mu_dists, single, notmu, state_only=False):
    inputs = [{} for _ in range(len(mu_dists))]
    for x in range(obs_shape[1]):
        x_onehot = onehot(x, obs_shape[1]).tolist()
        for y in range(obs_shape[0]):
            for t in range(horizon):
                for idx in range(len(mu_dists)):
                    xy_onehot = x_onehot + onehot(y, obs_shape[0]).tolist()
                    if single:
                        for i in range(len(mu_dists)):
                            xym_onehot = xy_onehot + [mu_dists[i][t, y, x]]
                            inputs[f'{x}-{y}-{t}-m-{i}'] = xym_onehot
                    elif notmu:
                        inputs[f'{x}-{y}-{t}'] = xy_onehot
                    else:
                        mu = [mu_dists[idx][t, y, x]]
                        #for pop in range(len(mu_dists)):
                        #    if pop!=idx:
                        #        mu.append(mu_dists[pop][t, y, x])
                        xym_onehot = xy_onehot + mu 
                        inputs[idx][f'{x}-{y}-{t}-m'] = xym_onehot
    return inputs

def true_reward(pos, densities):
    eps = 1e-25
    goal_pos = _DEFAULT_GOAL_POSITION

    if _MODE=="Predator-Prey":
        r_mu = -1.0 * np.log(densities + eps) + 10 * np.dot(reward_matrix, densities)
        rew = r_mu
    else:
        r_mu = -1.0 * np.log(densities + eps) + np.dot(reward_matrix, densities)
        r_xy = np.array([-np.sum(np.abs(goal_pos[i] - pos)) for i in range(len(goal_pos))])
        rew = r_mu + r_xy
    return rew
