import cma
import numpy as np
from optimizer import BasicSGD, SGD, Adam


class ESBase(object):
    def __init__(self, num_params, pop_size):
        self.num_params = num_params
        self.pop_size = pop_size

        self.first_run = True
        self.sigma = None
        self.solutions = None
        self.best_reward = 0


    def ask(self):
        raise NotImplementedError

    def tell(self, reward_list):
        raise NotImplementedError

    def result(self):
        return self.get_best_params(), self.best_reward, self.curr_best_reward, np.mean(self.sigma)

    def get_curr_params(self):
        raise NotImplementedError

    @staticmethod
    def compute_weight_decay(decay_rate, model_param_list):
        model_param_grid = np.array(model_param_list)
        return -decay_rate * np.mean(model_param_grid * model_param_grid, axis=1)

    @staticmethod
    def compute_centered_ranks(x):
        rx = x.ravel()
        ranks = np.empty(len(rx), dtype=int)
        ranks[rx.argsort()] = np.arange(len(rx))
        y = ranks.reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= 0.5
        return y


class GA(ESBase):

    def __init__(self, num_params,
                 sigma_init=0.1,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 pop_size=256,
                 elite_ratio=0.1,
                 forget_best=False,
                 weight_decay=0.01):
        ESBase.__init__(self, num_params, pop_size)
        self.sigma = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

        self.forget_best = forget_best
        self.weight_decay = weight_decay

        self.elite_size = int(self.pop_size * elite_ratio)
        self.elite_params = np.zeros((self.elite_size, self.num_params))
        self.elite_rewards = np.zeros(self.elite_size)

        self.best_params = np.zeros(self.num_params)

    def ask(self):
        self.epsilon = np.random.randn(self.pop_size, self.num_params) * self.sigma
        solutions = []

        elite_range = range(self.elite_size)
        for i in range(self.pop_size):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = self.mate(self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        assert len(reward_table_result) == self.pop_size, 'Inconsistent reward_table size'
        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            l2_decay = self.compute_weight_decay(self.weight_decay, self.solutions)
            # reward_table += l2_decay
            l2_decay = l2_decay.reshape(self.pop_size, 1)
            self.solutions += l2_decay

        if self.first_run or self.forget_best:
            reward = reward_table
            solution = self.solutions
            self.first_run = False
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_size]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_run or self.curr_best_reward > self.best_reward:
            self.first_run = False
            self.best_reward = self.elite_rewards[0]
            self.best_params = np.copy(self.elite_params[0])

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    def get_curr_params(self):
        return self.elite_params[0]

    def get_best_params(self):
        return self.best_params

    @staticmethod
    def mate(a, b):
        c = np.copy(a)
        idx = np.where(np.random.rand((c.size)) > 0.5)
        c[idx] = b[idx]
        return c


class CMAES(ESBase):

    def __init__(self, num_params,
                 sigma_init=0.10,
                 pop_size=256,
                 weight_decay=0.01):
        ESBase.__init__(self, num_params, pop_size)
        self.weight_decay = weight_decay

        self.es = cma.CMAEvolutionStrategy(
            self.num_params * [0], sigma_init, {'popsize': self.pop_size})

    def ask(self):
        solutions = np.array(self.es.ask())
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result)  # WHY??
        if self.weight_decay > 0:
            l2_decay = self.compute_weight_decay(self.weight_decay, self.solutions)
            # reward_table += l2_decay
            l2_decay = l2_decay.reshape(self.pop_size, 1)
            self.solutions += l2_decay

        self.es.tell(self.solutions, (reward_table).tolist())

    def result(self):
        # Overwrite result() function.
        r = self.es.result
        return (r[0], -r[1], -r[1], np.mean(r[6]))

    def get_curr_params(self):
        return self.es.result[5]

    def get_best_params(self):
        return self.es.result[0]


class PEPG(ESBase):

    def __init__(self, num_params,
                 sigma_init=0.10,
                 sigma_alpha=0.20,  # learning rate for std
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 sigma_max_change=0.2,  # clip adaptive sigma to 0.2
                 lr=0.01,
                 lr_decay=0.9999,
                 lr_limit=0.01,
                 elite_ratio=0,  # if > 0, then ignore lr
                 pop_size=256,
                 average_baseline=True,  # set baseline to average of batch
                 weight_decay=0.01,
                 rank_fitness=True,
                 forget_best=True):  # don't keep the historical best solution
        ESBase.__init__(self, num_params, pop_size)
        self.sigma = np.ones(self.num_params) * sigma_init
        self.sigma_alpha = sigma_alpha
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma_max_change = sigma_max_change

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit

        self.average_baseline = average_baseline
        if self.average_baseline:
            assert self.pop_size % 2 == 0, 'Population size must be even.'
            self.batch_size = int(self.pop_size / 2)
        else:
            assert self.pop_size % 2 == 1, 'Population size must be odd'
            self.batch_size = int((self.pop_size - 1) / 2)

        self.elite_size = int(self.pop_size * elite_ratio)
        self.use_elite = True if self.elite_size > 0 else False

        self.forget_best = forget_best
        self.batch_reward = np.zeros(self.batch_size * 2)
        self.mu = np.zeros(self.num_params)

        self.curr_best_mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)

        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True

        self.optimizer = Adam(num_params)

    def ask(self):
        self.epsilon = np.random.randn(self.batch_size, self.num_params) * \
            self.sigma.reshape(1, self.num_params)
        self.epsilon_full = np.concatenate([self.epsilon, -self.epsilon])
        if self.average_baseline:
            epsilon = self.epsilon_full
        else:
            epsilon = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])
        solutions = self.mu.reshape(1, self.num_params) + epsilon
        self.solutions = solutions
        return solutions

    def tell(self, reward_table_result):
        assert len(reward_table_result) == self.pop_size, 'Inconsistent reward_table size.'
        reward_table = np.array(reward_table_result)

        if self.rank_fitness:
            reward_table = self.compute_centered_ranks(reward_table)

        if self.weight_decay:
            l2_decay = self.compute_weight_decay(self.weight_decay, self.solutions)
            # reward_table += l2_decay
            l2_decay = l2_decay.reshape(self.pop_size, 1)
            self.solutions += l2_decay

        if self.average_baseline:
            b = np.mean(reward_table)
            reward_offset = 0
        else:
            b = reward_table[0]  # baseline
            reward_offset = 1

        reward = reward_table[reward_offset:]
        if self.use_elite:
            idx = np.argsort(reward)[::-1][0:self.elite_size]
        else:
            idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        if best_reward > b or self.average_baseline:
            self.curr_best_mu = self.mu + self.epsilon_full[idx[0]]
            best_reward = reward[idx[0]]
        else:
            self.curr_best_mu = self.mu
            best_reward = b
        self.curr_best_reward = best_reward

        if self.first_run:
            self.first_run = False
            self.best_reward = self.curr_best_reward
            self.best_mu = self.curr_best_mu
        else:
            if self.forget_best or self.curr_best_reward > self.best_reward:
                self.best_mu = self.curr_best_mu
                self.best_reward = self.curr_best_reward

        epsilon = self.epsilon
        sigma = self.sigma

        # gradients of mu
        if self.use_elite:
            self.mu += self.epsilon_full[idx].mean(axis=0)
        else:
            rT = reward[:self.batch_size] - reward[self.batch_size:]
            change_mu = np.dot(rT, epsilon)
            step, infos = self.optimizer.compute_step(-change_mu, self.lr)
            self.mu += step

        # gradients of sigma
        if self.sigma_alpha > 0:
            stdev_reward = 1.0
            if not self.rank_fitness:
                stdev_reward = reward.std()
            S = (epsilon * epsilon - (sigma * sigma).reshape(1, self.num_params)) / \
                sigma.reshape(1, self.num_params)
            reward_avg = (reward[:self.batch_size] + reward[self.batch_size:]) / 2.0
            rS = reward_avg - b
            delta_sigma = np.dot(rS, S) / (2 * self.batch_size * stdev_reward)

            change_sigma = self.sigma_alpha * delta_sigma
            change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
            change_sigma = np.maximum(change_sigma, -self.sigma_max_change * self.sigma)
            self.sigma += change_sigma

        if self.sigma_decay < 1:
            self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay

        if self.lr_decay < 1 and self.lr > self.lr_limit:
            self.lr *= self.lr_decay

    def get_curr_params(self):
        return self.curr_best_mu

    def get_best_params(self):
        return self.best_mu


class OpenES(ESBase):

    def __init__(self, num_params,
                 sigma_init=0.1,
                 sigma_decay=0.999,
                 sigma_limit=0.01,
                 lr=0.01,
                 lr_decay=0.9999,
                 lr_limit=0.001,
                 pop_size=256,
                 antithetic=True,
                 weight_decay=0.01,
                 rank_fitness=True,
                 forget_best=True):
        ESBase.__init__(self, num_params, pop_size)
        self.sigma = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_limit = lr_limit

        self.antithetic = antithetic
        if self.antithetic:
            assert self.pop_size % 2 == 0, 'Population size must be even'
            self.half_popsize = int(self.pop_size / 2)

        self.reward = np.zeros(self.pop_size)
        self.mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)

        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True

        self.optimizer = Adam(num_params)

    def ask(self):
        if self.antithetic:
            self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
            self.epsilon = np.concatenate([self.epsilon_half, -self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.pop_size, self.num_params)

        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, reward_table_result):
        assert len(reward_table_result) == self.pop_size, 'Inconsistent reward_table size reported.'
        reward = np.array(reward_table_result)

        if self.rank_fitness:
            reward = self.compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = self.compute_weight_decay(self.weight_decay, self.solutions)
            reward += l2_decay

        idx = np.argsort(reward)[::-1]

        self.curr_best_reward = reward[idx[0]]
        self.curr_best_mu = self.solutions[idx[0]]

        if self.first_run:
            self.first_run = False
            self.best_reward = self.curr_best_reward
            self.best_mu = self.curr_best_mu
        else:
            if self.forget_best or self.curr_best_reward > self.best_reward:
                self.best_mu = self.curr_best_mu
                self.best_reward = self.curr_best_reward

        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1.0 / (self.pop_size * self.sigma) * np.dot(self.epsilon.T, normalized_reward)

        step, infos = self.optimizer.compute_step(-change_mu, self.lr)
        self.mu += step

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

        if self.lr > self.lr_limit:
            self.lr *= self.lr_decay

    def get_curr_params(self):
        return self.curr_best_mu

    def get_best_params(self):
        return self.best_mu


def make_es(args, pop_size, num_params):
    if args.es_name == 'ses':
        es = PEPG(
            num_params,
            sigma_init=args.sigma_init,
            sigma_decay=args.sigma_decay,
            sigma_alpha=0.2,
            sigma_limit=0.02,
            elite_ratio=0.1,
            weight_decay=0.005,
            pop_size=pop_size)
    elif args.es_name == 'ga':
        es = GA(
            num_params,
            sigma_init=args.sigma_init,
            sigma_decay=args.sigma_decay,
            sigma_limit=0.02,
            elite_ratio=0.1,
            weight_decay=0.005,
            pop_size=pop_size)
    elif args.es_name == 'cma':
        es = CMAES(
            num_params,
            sigma_init=args.sigma_init,
            pop_size=pop_size)
    elif args.es_name == 'pepg':
        es = PEPG(
            num_params,
            sigma_init=args.sigma_init,
            sigma_decay=args.sigma_decay,
            sigma_alpha=0.2,
            sigma_limit=0.02,
            lr=0.01,
            lr_decay=1.0,
            lr_limit=0.01,
            weight_decay=0.005,
            pop_size=pop_size)
    elif args.es_name == 'openes':
        es = OpenES(
            num_params,
            sigma_init=args.sigma_init,
            sigma_decay=args.sigma_decay,
            sigma_limit=0.02,
            lr=0.01,
            lr_decay=1.0,
            lr_limit=0.01,
            antithetic=args.antithetic,
            weight_decay=0.005,
            pop_size=pop_size)
    else:
        raise ValueError('Invalid ES name. (ses|pepg|openes|ga|cma)')
    return es
