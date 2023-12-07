import os
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse
from distutils.util import strtobool
import time
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from Multi_type_render import render
from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import rl_agent
from open_spiel.python import rl_agent_policy
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
import pyspiel
import logger



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="Set the name of this experiment")
    parser.add_argument("--game-name", type=str, default="python_mfg_predator_prey", help="Set the game name")
    parser.add_argument("--game-setting", type=str, default="", help="")
    parser.add_argument('--logdir', type=str, default="/mnt/shunsuke/result", help="logdir")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer")
    parser.add_argument("--num-episodes", type=int, default=5, help="set the number of episodes of to collect per rollout")
    parser.add_argument("--update-episodes", type=int, default=20, help="set the number of episodes of the inner loop")
    parser.add_argument("--update-iterations", type=int, default=100, help="Set the number of global update steps of the outer loop")
    
    parser.add_argument('--optimizer', type=str, default="Adam", help="Set the optimizer (Adam) or (SGD)")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="Use to repreduce experiment results")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help="Use Gpu to run the experiment")


    
    # PPO parameters
    parser.add_argument('--gamma', type=float, default=0.9, help='set discount factor gamma')
    parser.add_argument("--num-minibatches", type=int, default=5,  help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5, help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--max-grad-norm", type=float, default=5, help="the maximum norm for the gradient clipping")


    # MFPPO parameters
    parser.add_argument('--alpha', type= int, default=0.5, help='Set alpha to controll the iteration and epsiode policy updates')
    parser.add_argument('--eps-eps', type= int, default=0.2, help='eps to update the episode learned policy')
    parser.add_argument('--itr-eps', type= int, default=0.05, help='eps to update the episode learned policy')

    args = parser.parse_args()

    return args


class NashC(NashConv):
    # Mainly used to calculate the exploitability 
    def __init__(self, game,distrib,pi_value, root_state=None):
        self._game = game
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
            
        self._distrib = distrib

        self._pi_value = pi_value

        self._br_value = best_response_value.BestResponse(
            self._game,
            self._distrib,
            value.TabularValueFunction(self._game),
            root_state=root_state)


class Agent(nn.Module):
    def __init__(self, info_state_size, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.info_state_size = info_state_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(info_state_size, 128)), 
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,1))
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(info_state_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, num_actions))
        )
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, bias_const=0.0):
    # used to initalize layers
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOpolicy(policy_std.Policy):
    # required obeject to work with openspiel 
    # used in updating the distribution using the policy 
    # and in calculating the nash-convergance 

    def __init__(self, game, agent, player_ids, device):
        super().__init__(game, player_ids)
        self.agent = agent
        self.device = device

    def action_probabilities(self, state, player_id=None):
        # main method that is called to update the population states distribution
        obs = torch.Tensor(state.observation_tensor()).to(self.device)
        legal_actions = state.legal_actions()
        logits = self.agent.actor(obs).detach().cpu()
        legat_logits = np.array([logits[action] for action in legal_actions])
        probs = np.exp(legat_logits -legat_logits.max())
        probs /= probs.sum(axis=0)
        
        # returns a dictionary with actions as keys and their probabilities as values
        return {action:probs[legal_actions.index(action)] for action in legal_actions}


def rollout(envs, iter_agents, eps_agents, conv_dist, num_epsiodes, steps, device):
    # generates num_epsiodes rollouts
    num_agent = len(envs)

    info_state = [torch.zeros((steps,iter_agents[i].info_state_size), device=device) for i in range(num_agent)]
    obs_mu = [torch.zeros((steps, iter_agents[i].info_state_size+3), device=device) for i in range(num_agent)]
    actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    rewards = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    dones = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    values = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    entropies = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    t_actions = [torch.zeros((steps,), device=device) for _ in range(num_agent)]
    t_logprobs = [torch.zeros((steps,), device=device) for _ in range(num_agent)]

    size = envs[0].game.get_parameters()['size']
    step = 0
    for _ in range(num_epsiodes):
        time_steps = [envs[i].reset() for i in range(num_agent)]
        while not time_steps[0].last():
            mu = []
            for i in range(num_agent):
                obs = time_steps[i].observations["info_state"][0]
                obs = torch.Tensor(obs).to(device)
                info_state[i][step] = obs
                with torch.no_grad():
                    t_action, t_logprob, _, _ = iter_agents[i].get_action_and_value(obs)
                    action, logprob, entropy, value = eps_agents[i].get_action_and_value(obs)

                # iteration policy data
                t_logprobs[i][step] = t_logprob
                t_actions[i][step] = t_action
                logprobs[i][step] = logprob
                entropies[i][step] = entropy
                values[i][step] = value
                actions[i][step] = action

                time_steps[i] = envs[i].step([action.item()])

                obs_list = list(obs.detach().numpy())
                obs_x = obs_list[:size].index(1)
                obs_y = obs_list[size:2*size].index(1)
                obs_t = obs_list[2*size:].index(1)
                mu.append(conv_dist[i][obs_t, obs_y, obs_x])

            for i in range(num_agent):
                # episode policy data
                dones[i][step] = time_steps[i].last()
                rewards[i][step] = torch.Tensor(np.array(time_steps[i].rewards[i])).to(device)
            ob_mu = obs_list 
            ob_mu += mu
            obs_mu[i][step] = torch.Tensor(ob_mu).to(device)
            step += 1

    return info_state, obs_mu, actions, logprobs, rewards, dones, values, entropies,t_actions,t_logprobs 

def cal_Adv(gamma, norm, rewards,values, dones):
    # function used to calculate the Generalized Advantage estimate
    # using the exact method in stable-baseline3
    with torch.no_grad():
        next_done = dones[-1]
        next_value = values[-1] 
        steps = len(values)
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(steps)):
            if t == steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return

        advantages = returns - values

    if norm:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def update(update_epochs, num_minibatch, obs, logprobs, actions, advantages, returns, t_actions, t_logprobs, optimizer_actor, optimize_critic, agent, alpha = 0.5, t_eps = 0.2, eps = 0.2):
    # update the agent network (actor and critic)
    batch_size = actions.shape[0]
    b_inds = np.arange(batch_size)
    mini_batch_size = batch_size // num_minibatch
    # get batch indices
    np.random.shuffle(b_inds)
    for _ in range(update_epochs):
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = b_inds[start:end]
            # for each update epoch shuffle the batch indices
            # generate the new logprobs, entropy and value then calculate the ratio
            b_obs = obs[mb_inds]
            b_advantages = advantages[mb_inds]

            # Get the data under the episode policy (representative agent current policy)
            _, newlogprob, entropy, new_value = agent.get_action_and_value(b_obs, actions[mb_inds])
            logratio = newlogprob - logprobs[mb_inds]
            ratio = torch.exp(logratio)

            # Get the data under the iteration policy (the population policy)
            _, t_newlogprob, _, _ = agent.get_action_and_value(b_obs, t_actions[mb_inds])
            t_logratio = t_newlogprob - t_logprobs[mb_inds]
            t_ratio = torch.exp(t_logratio)

            # iteration update PPO
            t_pg_loss1 = b_advantages * t_ratio
            t_pg_loss2 = b_advantages * torch.clamp(t_ratio, 1 - t_eps, 1 + t_eps)
            
            # episodic update PPO 
            pg_loss1 = b_advantages * ratio
            pg_loss2 = b_advantages * torch.clamp(ratio, 1 - eps, 1 + eps)

            # Calculate the loss using our loss function 
            pg_loss = - alpha * torch.min(pg_loss1, pg_loss2).mean() - (1-alpha) * torch.min(t_pg_loss1, t_pg_loss2).mean()
            v_loss = F.smooth_l1_loss(new_value.reshape(-1), returns[mb_inds]).mean()
            entropy_loss = entropy.mean()

            loss = pg_loss - args.ent_coef * entropy_loss 
            
            # Actor update 
            optimizer_actor.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
            optimizer_actor.step()
            
            # Critic update 
            optimize_critic.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
            optimize_critic.step()

    return v_loss

def plot_dist(env, game_name, distrib, info_state, save=False, filename="agent_dist.mp4"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    horizon = env.game.get_parameters()['horizon']
    d_size = size = env.game.get_parameters()['size']
    agent_dist = np.zeros((horizon,d_size,d_size))
    mu_dist = np.zeros((horizon,d_size,d_size))


    for k,v in distrib.distribution.items():
        if "mu" in k:
            tt = k.split("_")[0].split(",")
            x = int(tt[0].split("(")[-1])
            y = int(tt[1].split()[-1])
            t = int(tt[2].split()[-1].split(")")[0])
            mu_dist[t,y,x] = v

    for i in range(horizon):
        obs = info_state[i].tolist()
        obs_x = obs[:size].index(1)
        obs_y = obs[size:2*size].index(1)
        obs_t = obs[2*size:].index(1)
        agent_dist[obs_t,obs_y,obs_x] = 0.02

    final_dist = agent_dist + mu_dist

    if save:
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(img, animated=True)] for img in final_dist]
        ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)

        ani.save(filename, fps=5)

        plt.close()

def log_metrics(it, distrib, policy, writer, reward, entropy):
    # this function is used to log the results to tensor board
    initial_states = game.new_initial_states()
    pi_value = policy_value.PolicyValue(game, distrib, policy, value.TabularValueFunction(game))
    m = {
        f"ppo_br/{state}": pi_value.eval_state(state)
        for state in initial_states
    }
    m["nash_conv_ppo"] = NashC(game, distrib, pi_value).nash_conv()
    #writer.add_scalar("initial_state_value", m['ppo_br/initial'], it)
    ## debug
    #writer.add_scalar("rewards", reward, it)
    #writer.add_scalar("entorpy", entropy, it)

    #writer.add_scalar("nash_conv_ppo", m['nash_conv_ppo'], it)
    #logger.debug(f"ppo_br: {m['ppo_br/initial']}, and nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
    #print(f"ppo_br: {m['ppo_br/initial']}, and nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
    print(f"nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
    return m["nash_conv_ppo"]
    
def convert_distrib(envs, distrib):
    env = envs[0]
    num_agent = len(envs)
    horizon = env.game.get_parameters()['horizon']
    size = env.game.get_parameters()['size']
    mu_dist = [np.zeros((horizon, size, size)) for _ in range(num_agent)]

    for k,v in distrib.distribution.items():
        if "mu" in k:
            tt = k.split(",")
            pop = int(tt[0][-1])
            t = int(tt[1][3])
            xy = tt[2].split(" ")
            x = int(xy[1].split("[")[-1])
            y = int(xy[2].split("]")[0])
            mu_dist[pop][t,y,x] = v
    return mu_dist

if __name__ == "__main__":
    args = parse_args()

    logger.configure(args.logdir, format_strs=['stdout', 'log', 'json'])

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    # choose a value for the best model 
    # lower than which we save the weights and distribution 
    best_model = 300
    
    # Set the device (in our experiments CPU vs GPU does not improve time at all) we recommend CPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Set the file name
    # fname = "result/maze_all_exp"
    fname = args.logdir
    
    # logging 
    run_name = f"{args.exp_name}_{args.game_setting}_{args.optimizer}_num_update_epochs_{args.update_epochs}_num_episodes_per_rollout_{args.num_episodes}_number_of_mini_batches_{args.num_minibatches}_{time.asctime(time.localtime(time.time()))}"
    log_name = os.path.join(fname, run_name)
    tb_writer = SummaryWriter(log_name)
    LOG = log_name + "_log.txt"                                                    
    logging.basicConfig(filename=LOG, filemode="a", level=logging.DEBUG, force=True)  

    # console handler  
    console = logging.StreamHandler()  
    console.setLevel(logging.ERROR)  
    logging.getLogger("").addHandler(console)
    

    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key,value in vars(args).items()])),
    )
    
    # Create the game instance 
    game = pyspiel.load_game('python_mfg_predator_prey')
    game.new_initial_state()

    num_agent = 3

    mfg_dists = []
    for i in range(num_agent):
        uniform_policy = policy_std.UniformRandomPolicy(game)
        mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
        mfg_dists.append(mfg_dist)
    merge_dist = distribution.MergeDistribution(game, mfg_dists)
    
    envs = []
    for i in range(num_agent):
        envs.append(rl_environment.Environment(game, mfg_distribution=merge_dist, mfg_population=i))
        envs[-1].seed(args.seed)

    info_state_size = envs[0].observation_spec()["info_state"][0]
    num_actions = envs[0].action_spec()["num_actions"]
    agents = []
    ppo_policies = []
    pop_agents = []
    optimizer_actors = []
    optimizer_critics = []
    mfg_dists = []
    for i in range(num_agent):
        # Creat the agent and population policies 
        agents.append(Agent(info_state_size, num_actions).to(device))
        ppo_policies.append(PPOpolicy(game, agents[-1], None, device))
        pop_agents.append(Agent(info_state_size, num_actions).to(device))

        if args.optimizer == "Adam":
            optimizer_actors.append(optim.Adam(agents[-1].actor.parameters(), lr=args.lr,eps=1e-5))
            optimizer_critics.append(optim.Adam(agents[-1].critic.parameters(), lr=args.lr,eps=1e-5))
        else:
            optimizer_actors.append(optim.SGD(agents[-1].actor.parameters(), lr=args.lr, momentum=0.9))
            optimizer_critics.append(optim.SGD(agents[-1].critic.parameters(), lr=args.lr, momentum=0.9))

        mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
        mfg_dists.append(mfg_dist)
    
    merge_dist = distribution.MergeDistribution(game, mfg_dists)
    conv_dist = convert_distrib(envs, merge_dist)
    for env in envs:
      env.update_mfg_distribution(merge_dist)


    # Used to log data for debugging
    steps = args.num_episodes * envs[0].max_game_length
    episode_entropy = [[] for _ in range(num_agent)]
    total_entropy = [[] for _ in range(num_agent)]
    Nash_con_vect = [[] for _ in range(num_agent)]

    eps_reward = [[] for _ in range(num_agent)]
    total_reward = [[] for _ in range(num_agent)]

    for k in range(args.update_iterations):
        v_loss = []
        for eps in range(args.update_episodes):
            # collect rollout data
            obs, obs_mu, actions, logprobs, rewards, dones, values, entropies, t_actions, t_logprobs\
                = rollout(envs, pop_agents, agents, conv_dist, args.num_episodes, steps, device)
            #store rewards and entropy for debugging
            for i in range(num_agent):
                episode_entropy[i].append(entropies[i].mean().item())
                eps_reward[i].append(rewards[i].sum().item()/args.num_episodes)
                # Calculate the advantage function 
                adv, returns = cal_Adv(args.gamma, True, rewards[i], values[i], dones[i])
                # Update the learned policy and report loss for debugging
                v_loss.append(update(args.update_epochs,args.num_minibatches, obs[i], logprobs[i], actions[i], adv, returns, t_actions[i], t_logprobs[i], optimizer_actors[i], optimizer_critics[i], agents[i], args.alpha, args.itr_eps, args.eps_eps))
            
                #collect and print the metrics
                total_reward[i].append(np.mean(eps_reward[i]))
                total_entropy[i].append(np.mean(episode_entropy[i]))

        mu_dists = []
        for i in range(num_agent):
        
            # Update the iteration policy with the new policy 
            pop_agents[i].load_state_dict(agents[i].state_dict())
            
            
            # calculate the exploitability 
            Nash_con_vect.append(log_metrics(k+1, merge_dist, ppo_policies[i], tb_writer, total_reward[i][-1], total_entropy[i][-1]))

            # update the environment distribution 
            mfg_dist = distribution.DistributionPolicy(game, ppo_policies[-1])
            mfg_dists.append(mfg_dist)

            logger.record_tabular(f"total_step {i}", v_loss[i].item())
            logger.record_tabular(f"num_episodes {i}", eps)
            logger.record_tabular(f"num_iteration {i}", k)
            logger.record_tabular(f"nash_conv {i}", Nash_con_vect[-1])
            logger.record_tabular(f"mean_reward {i}", total_reward[i][-1])
            logger.dump_tabular()
        
        merge_dist = distribution.MergeDistribution(game, mfg_dists)
        conv_dist = convert_distrib(envs, merge_dist)
        for env in envs:
          env.update_mfg_distribution(merge_dist)
   
    steps = args.num_episodes * env.max_game_length
    obs, obs_mu, actions, logprobs, rewards, dones, values, entropies, t_actions, t_logprobs = rollout(envs, pop_agents, agents, mfg_dists, 1, env.max_game_length, device)
    filename = os.path.join(fname, f"res.mp4")
    obs_np = [obs[i].detach().numpy() for i in range(num_agent)]
    render(envs, args.game_name, mfg_dists, obs_np, save=True, filename=filename)
        

for i in range(num_agent):
    if best_model >= Nash_con_vect[-1]:    
            #save the distribution and weights for further analysis 
            filename = os.path.join(fname, f"distribution_{i}_{run_name}.pkl")
            utils.save_parametric_distribution(mfg_dists[i], filename)   
            torch.save(agents[i].actor.state_dict(),fname + f"actor{i}.pth")
            torch.save(agents[i].critic.state_dict(),fname + f"critic{i}.pth")
