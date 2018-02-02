import argparse
from config import tasks
from model import Model


def main(args):
    game = tasks[args.game_name]

    seed = 0
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
        print('Seed: %d' % seed)

    model = Model(game)
    print('Number of parameters %d' % model.num_params)
    model.make_env(render=True)

    if args.model_path is not None:
        model.load_model(args.model_path)
    else:
        params = model.get_random_model_params(stdev=0.1)
        model.set_model_params(params)

    if args.final_mode:
        total_reward = 0.0
        np.random.seed(seed)
        model.env.seed(seed)

        for i in range(100):
            reward, steps_taken = model.simulate(
                train_mode=False, render_mode=False, num_ep=1)
            total_reward += reward[0]
        print('Seed: %d Avg reward: %.4f' % (seed, total_reward / 100))
    else:
        if args.record_video:
            model.env = Monitor(
                model.env, directory='/tmp/' + args.game_name,
                video_callable=lambda episode_id: True, write_upon_reset=True, force=True)
        while True:
            reward, timesteps = model.simulate(
                is_train=False, render_mode=args.is_render, num_ep=1)
            print('Avg reward: %.4f Avg timesteps: %.4f' % (reward, np.mean(timesteps) + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', type=str, help='robo_pendulum, robo_ant, robo_humanoid, etc.')
    parser.add_argument('--record_video', type=bool, default=False)
    parser.add_argument('--final_mode', type=bool, default=False)
    parser.add_argument('--is_render', type=bool, default=True)
    parser.add_argument('-m', '--model_path', type=str, default=None)
    parser.add_argument('-s', '--seed', type=int, default=111, help='initial seed')

    main(args)
