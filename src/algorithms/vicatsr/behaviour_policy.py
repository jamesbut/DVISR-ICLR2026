# Behaviour policy in order to do off-policy RL
# i.e. when the behaviour policy is different to the target policy.

from .equation import optimise_eq_consts


class BehaviourPolicy:

    def __init__(self, target_policy=None):

        # If target policy is provided, use as the behaviour policy
        self._target_policy = target_policy

    def sample(self):

        if self._target_policy is not None:
            return self._target_policy.sample()

        else:
            pass

    def sample_and_optimise(self, data, log_likelihood_func):
        return optimise_eq_consts(self.sample(), data, log_likelihood_func)
