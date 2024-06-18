# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for greedy_policy."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python import policy
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms import greedy_policy
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import crowd_modelling  # pylint: disable=unused-import
import pyspiel


class GreedyPolicyTest(parameterized.TestCase):

  @parameterized.named_parameters(('python', 'python_mfg_crowd_modelling'),
                                  ('cpp', 'mfg_crowd_modelling'))
  def test_greedy(self, name):
    """Check if the greedy policy works as expected.

    The test checks that a greedy policy with respect to an optimal value is
    an optimal policy.

    Args:
      name: Name of the game.
    """
    game = pyspiel.load_game(name)
    uniform_policy = policy.UniformRandomPolicy(game)
    dist = distribution.DistributionPolicy(game, uniform_policy)
    br_value = best_response_value.BestResponse(
        game, dist, value.TabularValueFunction(game))
    br_val = br_value(game.new_initial_state())

    greedy_pi = greedy_policy.GreedyPolicy(game, None, br_value)
    greedy_pi = greedy_pi.to_tabular()
    pybr_value = policy_value.PolicyValue(game, dist, greedy_pi,
                                          value.TabularValueFunction(game))
    pybr_val = pybr_value(game.new_initial_state())
    self.assertAlmostEqual(br_val, pybr_val)


if __name__ == '__main__':
  absltest.main()
