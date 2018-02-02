import os
import json
import numpy as np


class Seeder(object):
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2**31 - 1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result

    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result


class Logger(object):
    def __init__(self, args):
        log_dir = './log/' if args.log_dir is None else args.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        pop_size = args.num_workers * args.num_trails
        filebase = log_dir + args.game_name + '-' + args.es_name + '-' + str(pop_size)
        self.file = filebase + '.json'
        self.file_eval = filebase + '.eval.json'
        self.file_hist = filebase + '.hist.json'
        self.file_best = filebase + '.best.json'

        # print task info
        print('------------------------------')
        print('%-20s: %s' % ('Game', args.game_name))
        print('%-20s: %d' % ('Num_worker', args.num_workers))
        print('%-20s: %d' % ('Num_worker_trial', args.num_trails))
        print('%-20s: %d' % ('Population', args.num_workers * args.num_trails))
        print('------------------------------')

    def write_params(self, params):
        with open(self.file, 'wt') as out:
            res = json.dump(
                [np.array(params).round(4).tolist()], out,
                sort_keys=True, indent=2, separators=(',', ': '))

    def write_history(self, history):
        with open(self.file_hist, 'wt') as out:
            res = json.dump(
                history, out,
                sort_keys=False, indent=0, separators=(',', ': '))

    def write_eval(self, eval_info):
        with open(self.file_eval, 'wt') as out:
            res = json.dump(eval_info, out)

    def write_best(self, best):
        with open(self.file_best, 'wt') as out:
            res = json.dump(
                best, out,
                sort_keys=True, indent=0, separators=(',', ': '))

    def log_gen(self, info):
        print('gen:%-4d time:%-4d avg|min|max|std: %-8.2f %-6.2f %-8.2f %-6.2f avg_steps: %-6.1f avg_sigma: %-6.4f' % info)

    def log_eval(self, info):
        print('EVAL gen:%-4d curr_rew:%-8.2f improvement: %-8.2f best_rew: %-8.2f' % info)


class Communicator(object):
    ''' A class that manage communication between master and workers. '''

    def __init__(self, comm, args, num_params):
        self._comm = comm
        self.num_workers = args.num_workers
        self.master_rank = 0
        self.worker_ranks = range(1, self.num_workers+1)

        self.num_trails = args.num_trails
        self.pop_size = args.num_workers * args.num_trails

        self.precision = 10000  # packaged into a int array
        self.solution_packet_size = (5 + num_params) * args.num_trails
        self.result_packet_size = 4 * args.num_trails


class MasterComm(Communicator):
    def __init__(self, comm, args, num_params):
        Communicator.__init__(self, comm, args, num_params)

    def _encode_solution(self, seeds, solutions, is_train, max_len=-1):
        n = len(seeds)
        result = []
        worker_num = 0
        train_mode = 1 if is_train else 0
        for i in range(n):
            worker_num = int(i / self.num_trails) + 1
            result.append([worker_num, i, seeds[i], train_mode, max_len])
            result.append(np.round(np.array(solutions[i]) * self.precision, 0))
        result = np.concatenate(result).astype(np.int32)
        result = np.split(result, self.num_workers)
        return result

    def distribute_solutions(self, seeds, solutions, is_train=True, max_len=-1):
        solution_list = self._encode_solution(seeds, solutions, is_train, max_len=max_len)
        assert len(solution_list) == self.num_workers
        for i in self.worker_ranks:
            packet = solution_list[i - 1]
            assert len(packet) == self.solution_packet_size
            self._comm.Send(packet, dest=i)

    def gather_results(self):
        result_packet = np.empty(self.result_packet_size, dtype=np.int32)
        reward_list_total = np.zeros((self.pop_size, 2))
        check_results = np.ones(self.pop_size, dtype=np.int)
        for i in self.worker_ranks:
            self._comm.Recv(result_packet, source=i)
            results = self._decode_result(result_packet)
            for result in results:
                worker_id = int(result[0])
                assert worker_id == i, 'work_id=%d source=%d' % (worker_id, i)
                idx = int(result[1])
                reward_list_total[idx, 0] = result[2]
                reward_list_total[idx, 1] = result[3]
                check_results[idx] = 0

        check_sum = check_results.sum()
        assert check_sum == 0, check_sum
        return reward_list_total

    def _decode_result(self, packet):
        r = packet.reshape(self.num_trails, 4)
        workers = r[:, 0].tolist()
        jobs = r[:, 1].tolist()
        fits = r[:, 2].astype(np.float) / self.precision
        fits = fits.tolist()
        times = r[:, 3].astype(np.float) / self.precision
        times = times.tolist()
        result = []
        for i in range(len(jobs)):
            result.append([workers[i], jobs[i], fits[i], times[i]])
        return result


class WorkerComm(Communicator):
    def __init__(self, comm, args, num_params):
        Communicator.__init__(self, comm, args, num_params)

    def receive_solution(self):
        solution_packet = np.empty(self.solution_packet_size, dtype=np.int32)
        self._comm.Recv(solution_packet, source=self.master_rank)
        assert len(solution_packet) == self.solution_packet_size
        solutions = self._decode_solution(solution_packet)
        return solutions

    def _decode_solution(self, packet):
        packets = np.split(packet, self.num_trails)
        result = []
        for p in packets:
            result.append(
                [int(p[0]), int(p[1]), int(p[2]), p[3] == 1,
                 p[4], p[5:].astype(np.float) / self.precision])
        return result

    def send_results(self, results):
        result_packet = self._encode_result(results)
        assert len(result_packet) == self.result_packet_size
        self._comm.Send(result_packet, dest=self.master_rank)

    def _encode_result(self, results):
        r = np.array(results)
        r[:, 2:4] *= self.precision
        return r.flatten().astype(np.int32)


class ActivationFunc(object):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def passthru(x):
        return x

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def sample(p):
        return np.argmax(np.random.multinomial(1, p))

    @staticmethod
    def tanh(x):
        return np.tanh(x)


af = {'relu': ActivationFunc.sigmoid,
      'sigmoid': ActivationFunc.sigmoid,
      'passthru': ActivationFunc.passthru,
      'softmax': ActivationFunc.softmax,
      'tanh': ActivationFunc.tanh,
      'sample': ActivationFunc.sample}
