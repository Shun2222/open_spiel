import os
from typing import Sequence

from absl import flags

from open_spiel.python.mfg import utils
from open_spiel.python.mfg.algorithms import fictitious_play
from open_spiel.python.mfg.algorithms import nash_conv
from open_spiel.python.mfg.games import factory
from open_spiel.python.utils import app
from open_spiel.python.utils import metrics

from open_spiel.python.mfg import value
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import policy_value

from open_spiel.python import rl_environment
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.nfg.examples import mfg_Proximal_policy_optimization_pytorch

class TestGame(object):
    def __init__(self, game:pyspiel.Game)
        self._game = game
        self._policy = policy_std.UniformRandomPolicy(self._game)

    def iteration(self):
        distrib = distribution.DistributionPolicy(self._game, self._policy)
        player_ids = list(range(self._game.num_players()))
        br_value = best_response_value.BestResponse(
            self._game, distrib, value.TabularValueFunction(self._game))
        pi = greedy_policy.GreedyPolicy(self._game, player_ids, br_value)

    def get_policy(self):
        return self._policy

class TestEnv(object):
    def __init__(self):
        self._device = torch.device("cpu")


    def rollout(self, env, iter_agent, eps_agent):
        steps = env.max_game_length
        info_state = torch.zeros((steps,iter_agent.info_state_size), device=self._device)
        actions = torch.zeros((steps,), device=self._device)
        logprobs = torch.zeros((steps,), device=self._device)
        rewards = torch.zeros((steps,), device=self._device)
        dones = torch.zeros((steps,), device=self._device)
        values = torch.zeros((steps,), device=self._device)
        entropies = torch.zeros((steps,), device=self._device)
        t_actions = torch.zeros((steps,), device=self._device)
        t_logprobs = torch.zeros((steps,), device=self._device)

        step = 0
        time_step = env.reset()
        while not time_step.last():
            obs = time_step.observations["info_state"][0]
            obs = torch.Tensor(obs).to(self._device)
            info_state[step] = obs
            with torch.no_grad():
                t_action, t_logprob, _, _ = iter_agent.get_action_and_value(obs)
                action, logprob, entropy, value = eps_agent.get_action_and_value(obs)
            time_step = env.step([action.item()])

            t_logprobs[step] = t_logprob
            t_actions[step] = t_action

            logprobs[step] = logprob
            dones[step] = time_step.last()
            entropies[step] = entropy
            values[step] = value
            actions[step] = action
            rewards[step] = torch.Tensor(time_step.rewards).to(self._device)
            step += 1
        return info_state, actions, logprobs, rewards, dones, values, entropies,t_actions,t_logprobs 

    def test(self):
        game = factory.create_game_with_setting("mfg_crowd_modelling_2d", "crowd_modelling_2d_four_rooms") # crowd_modelling_2d_four_rooms or crowd_modelling_2d_maze
        uniform_policy = policy_std.UniformRandomPolicy(game)
        mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
        env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)
        env.seed(0)

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]
        agent = Agent(info_state_size,num_actions).to(self._device)
        ppo_policy = PPOpolicy(game, agent, None, self._device)
        pop_agent = Agent(info_state_size,num_actions).to(self._device)
        
        obs, actions, logprobs, rewards, dones, values, entropies, t_actions, t_logprobs =          self.rollout(env, pop_agent, agent)

        print(f"------------
                obs:{obs}\n
                actions:{actions}\n
                logprobs:{logprobs}\n
                rewards:{rewards}\n
                dones:{dones}\n
                values:{values}\n
                entropies:{entropies}\n
                t_actions:{t_actions}\n
                t_logprobs:{t_logprobs}\n
                ------------
             ")


def main(argv: Sequence[str]) -> None:
    testEnv = TestEnv()
    testEnv.test()


if __name__ == '__main__':
  app.run(main)
