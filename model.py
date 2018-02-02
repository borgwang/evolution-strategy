import json
import numpy as np

from gym.wrappers import Monitor
from utils import af
from config import make_env


class Model(object):

    def __init__(self, task):
        self.env_name = task.env_name
        self.time_input = 0  # use extra sinusoid input
        self.sigma_bias = task.noise_bias  # bias in stdev of output
        self.sigma_factor = 0.5  # multiplicative in stdev of output
        self.output_noise = task.output_noise

        if task.time_factor > 0:
            self.time_factor = float(task.time_factor)
            self.time_input = 1

        self.net_shapes = [
            (task.input_size + self.time_input, task.layers[0]),
            (task.layers[0], task.layers[1]),
            (task.layers[1], task.output_size)]

        self.net_atv = [af[a] for a in task.activation]
        self._build_net()

    def _build_net(self):
        self.weight = []
        self.bias = []
        self.bias_log_std = []
        self.bias_std = []

        self.num_params = 0

        for idx, shape in enumerate(self.net_shapes):
            self.weight.append(np.zeros(shape=shape))
            self.bias.append(np.zeros(shape=shape[1]))
            self.num_params += (np.product(shape) + shape[1])
            if self.output_noise[idx]:
                self.num_params += shape[1]
            log_std = np.zeros(shape=shape[1])
            self.bias_log_std.append(log_std)
            out_std = np.exp(self.sigma_factor * log_std + self.sigma_bias)
            self.bias_std.append(out_std)

    def make_env(self, seed=-1, render=False):
        self.env = make_env(self.env_name, seed=seed, render=render)

    def get_action(self, x, t=0):
        # feed forward through the network;;
        out = np.array(x).flatten()
        if self.time_input == 1:
            time_signal = float(t) / self.time_factor
            out = np.concatenate([out, [time_signal]])
        num_layers = len(self.weight)
        for i in range(num_layers):
            w = self.weight[i]
            b = self.bias[i]
            out = np.matmul(out, w) + b
            if self.output_noise[i]:
                out_size = self.net_shapes[i][1]
                out_std = self.bias_std[i]
                output_noise = np.random.randn(out_size) * out_std
                out += output_noise
            out = self.net_atv[i](out)

        return out

    def get_random_model_params(self, stdev=0.1):
        return np.random.randn(self.num_params) * stdev

    def set_model_params(self, model_params):
        pointer = 0
        for i in range(len(self.net_shapes)):
            w_shape = self.net_shapes[i]
            b_shape = self.net_shapes[i][1]
            s_w = np.product(w_shape)
            s = s_w + b_shape
            chunk = np.array(model_params[pointer: pointer + s])
            self.weight[i] = chunk[:s_w].reshape(w_shape)
            self.bias[i] = chunk[s_w:].reshape(b_shape)
            pointer += s
            if self.output_noise[i]:
                s = b_shape
                self.bias_log_std[i] = np.array(model_params[pointer: pointer + s])
                self.bias_std[i] = np.exp(
                    self.sigma_factor * self.bias_log_std[i] + self.sigma_bias)
                pointer += s

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print('loading file %s' % filename)
        model_params = np.array(data[0])
        self.set_model_params(model_params)

    def simulate(self, is_train=False, render_mode=True, num_ep=5, seed=-1, max_len=-1):
        reward_list, t_list = [], []
        orig_mode = True
        max_episode_length = 3000

        if is_train and max_len > 0:
            if max_len < max_episode_length:
                max_episode_length = max_len

        if seed >= 0:
            np.random.seed(seed)
            self.env.seed(seed)

        for _ in range(num_ep):
            reward_threshold = 300
            ep_reward = 0.0
            stumbled = False

            obs = self.env.reset()
            for t in range(max_episode_length):
                if render_mode:
                    self.env.render('human')
                action = self.get_action(obs, t=t)
                prev_obs = obs
                obs, reward, done, info = self.env.step(action)

                if is_train and reward == -100 and (not orig_mode):
                    reward = 0
                    stumbled = True

                ep_reward += reward

                if done:
                    if is_train and not stumbled and ep_reward > reward_threshold and not orig_mode:
                        ep_reward += 100
                    break

            reward_list.append(ep_reward)
            t_list.append(t)
            if render_mode:
                print('reward: %.2f timesteps: %.2f' % (ep_reward, t))

        return reward_list, t_list
