from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys
import time
import platform
import argparse

from model import Model
from es import make_es
from utils import Seeder, MasterComm, WorkerComm, Logger
from config import tasks


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class Master(object):

    def __init__(self):
        task = tasks[args.game_name]
        self.model = Model(task)
        self.model.make_env()
        num_params = self.model.num_params
        print('Number of parameters: %d' % num_params)
        self.logger = Logger(args)
        self.pop_size = args.num_workers * args.num_trails
        self.es = make_es(args, self.pop_size, num_params)
        self.seeder = Seeder(args.seed)
        self.comm = MasterComm(COMM, args, num_params)

    def run(self):
        start_time = int(time.time())
        history, eval_info = [], []

        i = 0
        while True:
            i += 1
            solutions = self.es.ask()
            seeds = self.get_seeds()
            self.comm.distribute_solutions(seeds, solutions)
            results = self.comm.gather_results()

            rewards, steps = results[:, 0], results[:, 1]

            self.es.tell(rewards)

            best_params, best_reward, curr_reward, sigma = self.es.result()

            self.model.set_model_params(np.array(best_params).round(4))

            curr_time = int(time.time()) - start_time

            info = (i, curr_time, np.mean(rewards), np.min(rewards),
                    np.max(rewards), np.std(rewards), np.mean(steps), sigma)

            self.logger.write_params(self.es.get_curr_params())
            self.logger.log_gen(info)
            history.append(info)
            self.logger.write_history(history)

            if i == 1:
                best_reward_eval = np.mean(rewards)
            if i % args.eval_interal == 0:
                curr_params = np.array(self.es.get_curr_params().round(4))
                reward_eval = self.eval_batch(curr_params)
                curr_params = curr_params.tolist()
                improvement = reward_eval - best_reward_eval
                eval_info.append([i, reward_eval, curr_params])
                self.logger.write_eval(eval_info)

                if len(eval_info) == 1 or reward_eval > best_reward_eval:
                    best_reward_eval = reward_eval
                    best_params_eval = curr_params

                self.logger.write_best([best_params_eval, best_reward_eval])
                eval_log = (i, reward_eval, improvement, best_reward_eval)
                self.logger.log_eval(eval_log)

    def eval_batch(self, model_params):
        solutions = []
        for i in range(self.pop_size):
            solutions.append(np.copy(model_params))
        seeds = np.arange(self.pop_size)

        self.comm.distribute_solutions(
            seeds, solutions, is_train=False, max_len=-1)

        results = self.comm.gather_results()
        rewards = results[:, 0]
        return np.mean(rewards)

    def get_seeds(self):
        if args.antithetic:
            seeds = self.seeder.next_batch(int(self.pop_size / 2))
            seeds = seeds + seeds
        else:
            seeds = self.seeder.next_batch(self.pop_size)
        return seeds


class Worker(object):
    def __init__(self):
        task = tasks[args.game_name]
        self.model = Model(task)
        self.model.make_env()
        num_params = self.model.num_params
        self.pop_size = args.num_workers * args.num_trails
        self.comm = WorkerComm(COMM, args, num_params)

    def run(self):
        while True:
            solutions = self.comm.receive_solution()
            results = []
            for solution in solutions:  # multi trails per worker
                worker_id, jobidx, seed, is_train, max_len, weights = solution
                assert worker_id == RANK, 'worker_id=%d rank=%d' % (worker_id, RANK)
                # simulate
                self.model.set_model_params(weights)
                rewards, timesteps = self.model.simulate(
                    is_train=is_train, render_mode=False,
                    num_ep=1, seed=seed, max_len=max_len)
                fitness = np.mean(rewards)
                timestep = np.mean(timesteps)

                results.append([worker_id, jobidx, fitness, timestep])
            self.comm.send_results(results)


def main():
    if RANK == 0:
        print('Master started. %d processes.' % SIZE)
        Master().run()
    else:
        print('Worker-%d started. %d processes.' % (RANK, SIZE))
        Worker().run()


def mpi_run():
    if os.getenv('IN_MPI') is None:
        # fork processes
        env = os.environ.copy()
        env.update(IN_MPI='1')
        mpi_cmd = ['mpirun', '-np', str(args.num_workers + 1)]
        script = [sys.executable, '-u'] + sys.argv
        if platform.system() == 'Darwin':  # for Mac
            mpi_cmd.extend(['--hostfile', 'hostfile'])
        cmd = mpi_cmd + script
        print('RUN: %s' % (' '.join(cmd)))
        subprocess.check_call(cmd, env=env)
        sys.exit()  # admin process exit
    else:
        main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', type=str, help='robo_pendulum, robo_ant, robo_humanoid, etc.')
    parser.add_argument('-e', '--es_name', type=str,
                        help='ses, pepg, openes, ga, cma.', default='cma')
    parser.add_argument('--eval_interal', type=int, default=25, help='evaluate every k generations')
    parser.add_argument('-n', '--num_workers', type=int, default=8)
    parser.add_argument('-t', '--num_trails', type=int, help='trials per worker', default=4)
    parser.add_argument('--antithetic', type=bool, default=True,
                        help='set to 0 to disable antithetic sampling')
    parser.add_argument('-s', '--seed', type=int, default=111, help='initial seed')
    parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
    parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')

    global args
    args = parser.parse_args()
    assert args.num_workers > 0, 'Number of workers suppose to > 0.'
    mpi_run()
