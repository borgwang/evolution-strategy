from collections import namedtuple

Task = namedtuple(
    'Task',
    ['env_name', 'time_factor', 'input_size', 'output_size', 'layers',
     'activation', 'noise_bias', 'output_noise'])

########## BULLET ENVS ##########
bullet_pendulum = Task(env_name='InvertedPendulumSwingupBulletEnv-v0',
                       input_size=5,
                       output_size=1,
                       time_factor=1000,
                       layers=[25, 5],
                       activation=['tanh', 'tanh', 'passthru'],
                       noise_bias=0.0,
                       output_noise=[False, False, True],
                       )

bullet_double_pendulum = Task(env_name='InvertedDoublePendulumBulletEnv-v0',
                              input_size=9,
                              output_size=1,
                              time_factor=0,
                              layers=[45, 5],
                              activation=['tanh', 'tanh', 'passthru'],
                              noise_bias=0.0,
                              output_noise=[False, False, True],
                              )

bullet_minitaur_duck = Task(env_name='MinitaurDuckBulletEnv-v0',
                            input_size=28,
                            output_size=8,
                            time_factor=0,
                            layers=[64, 32],
                            activation=['tanh', 'tanh', 'tanh'],
                            noise_bias=0.0,
                            output_noise=[False, False, False],
                            )

bullet_kuka_grasping = Task(env_name='KukaBulletEnv-v0',
                            input_size=9,
                            output_size=3,
                            time_factor=0,
                            layers=[64, 32],
                            activation=['tanh', 'tanh', 'tanh'],
                            noise_bias=0.0,
                            output_noise=[False, False, False],
                            )

bullet_kuka_grasping_stoc = Task(env_name='KukaBulletEnv-v0',
                                 input_size=9,
                                 output_size=3,
                                 time_factor=0,
                                 layers=[64, 32],
                                 activation=['tanh', 'tanh', 'tanh'],
                                 noise_bias=0.0,
                                 output_noise=[False, False, True],
                                 )

bullet_minitaur_duck_stoc = Task(env_name='MinitaurDuckBulletEnv-v0',
                                 input_size=28,
                                 output_size=8,
                                 time_factor=0,
                                 layers=[64, 32],
                                 activation=['tanh', 'tanh', 'tanh'],
                                 noise_bias=-1.0,
                                 output_noise=[True, True, True],
                                 )

bullet_minitaur_ball = Task(env_name='MinitaurBallBulletEnv-v0',
                            input_size=28,
                            output_size=8,
                            time_factor=0,
                            layers=[64, 32],
                            activation=['tanh', 'tanh', 'tanh'],
                            noise_bias=0.0,
                            output_noise=[False, False, False],
                            )

bullet_minitaur_ball_stoc = Task(env_name='MinitaurBallBulletEnv-v0',
                                 input_size=28,
                                 output_size=8,
                                 time_factor=0,
                                 layers=[64, 32],
                                 activation=['tanh', 'tanh', 'tanh'],
                                 noise_bias=-1.0,
                                 output_noise=[True, True, True],
                                 )

bullet_half_cheetah = Task(env_name='HalfCheetahBulletEnv-v0',
                           input_size=26,
                           output_size=6,
                           time_factor=0,
                           layers=[64, 32],
                           activation=['tanh', 'tanh', 'tanh'],
                           noise_bias=0.0,
                           output_noise=[False, False, False],
                           )

bullet_humanoid = Task(env_name='HumanoidBulletEnv-v0',
                       input_size=44,
                       output_size=17,
                       layers=[220, 85],
                       time_factor=1000,
                       activation=['tanh', 'tanh', 'passthru'],
                       noise_bias=0.0,
                       output_noise=[False, False, True],
                       )

bullet_ant = Task(env_name='AntBulletEnv-v0',
                  input_size=28,
                  output_size=8,
                  layers=[64, 32],
                  time_factor=1000,
                  activation=['tanh', 'tanh', 'tanh'],
                  noise_bias=0.0,
                  output_noise=[False, False, True],
                  )

bullet_walker = Task(env_name='Walker2DBulletEnv-v0',
                     input_size=22,
                     output_size=6,
                     time_factor=1000,
                     layers=[110, 30],
                     activation=['tanh', 'tanh', 'passthru'],
                     noise_bias=0.0,
                     output_noise=[False, False, True],
                     )

bullet_hopper = Task(env_name='HopperBulletEnv-v0',
                     input_size=15,
                     output_size=3,
                     layers=[75, 15],
                     time_factor=1000,
                     activation=['tanh', 'tanh', 'passthru'],
                     noise_bias=0.0,
                     output_noise=[False, False, True],
                     )

bullet_racecar = Task(env_name='RacecarBulletEnv-v0',
                      input_size=2,
                      output_size=2,
                      time_factor=1000,
                      layers=[20, 20],
                      activation=['tanh', 'tanh', 'passthru'],
                      noise_bias=0.0,
                      output_noise=[False, False, False],
                      )

bullet_minitaur = Task(env_name='MinitaurBulletEnv-v0',
                       input_size=28,
                       output_size=8,
                       time_factor=0,
                       layers=[64, 32],
                       activation=['tanh', 'tanh', 'tanh'],
                       noise_bias=0.0,
                       output_noise=[False, False, False],
                       )

bullet_minitaur_stoc = Task(env_name='MinitaurBulletEnv-v0',
                            input_size=28,
                            output_size=8,
                            time_factor=0,
                            layers=[64, 32],
                            activation=['tanh', 'tanh', 'tanh'],
                            noise_bias=0.0,
                            output_noise=[True, True, True],
                            )

########## OPENAI GYM ENVS ##########
bipedhard_stoc = Task(env_name='BipedalWalkerHardcore-v2',
                      input_size=24,
                      output_size=4,
                      time_factor=1000,
                      layers=[120, 20],
                      activation=['tanh', 'tanh', 'passthru'],
                      noise_bias=0.0,
                      output_noise=[True, True, True],
                      )

bipedhard = Task(env_name='BipedalWalkerHardcore-v2',
                 input_size=24,
                 output_size=4,
                 time_factor=0,
                 layers=[40, 40],
                 activation=['tanh', 'tanh', 'tanh'],
                 noise_bias=0.0,
                 output_noise=[False, False, False],
                 )

carracing = Task(env_name='CarRacing-v0',
                 input_size=64,
                 output_size=3,
                 time_factor=0,
                 layers=[40, 40],
                 activation=['tanh', 'tanh', 'tanh'],
                 noise_bias=0.0,
                 output_noise=[False, False, True],
                 )

########## OPENAI ROBOSCHOOL ENVS ##########
robo_reacher = Task(env_name='RoboschoolReacher-v1',
                    input_size=9,
                    output_size=2,
                    layers=[45, 10],
                    time_factor=1000,
                    activation=['tanh', 'tanh', 'passthru'],
                    noise_bias=0.0,
                    output_noise=[False, False, True],
                    )

robo_flagrun = Task(env_name='RoboschoolHumanoidFlagrunHarder-v1',
                    input_size=44,
                    output_size=17,
                    layers=[220, 85],
                    time_factor=1000,
                    activation=['tanh', 'tanh', 'passthru'],
                    noise_bias=0.0,
                    output_noise=[False, False, True],
                    )

robo_pendulum = Task(env_name='RoboschoolInvertedPendulumSwingup-v1',
                     input_size=5,
                     output_size=1,
                     time_factor=1000,
                     layers=[25, 5],
                     activation=['tanh', 'tanh', 'passthru'],
                     noise_bias=0.0,
                     output_noise=[False, False, True],
                     )

robo_double_pendulum = Task(env_name='RoboschoolInvertedDoublePendulum-v1',
                            input_size=9,
                            output_size=1,
                            time_factor=0,
                            layers=[45, 5],
                            activation=['tanh', 'tanh', 'passthru'],
                            noise_bias=0.0,
                            output_noise=[False, False, True],
                            )

robo_humanoid = Task(env_name='RoboschoolHumanoid-v1',
                     input_size=44,
                     output_size=17,
                     layers=[220, 85],
                     time_factor=1000,
                     activation=['tanh', 'tanh', 'passthru'],
                     noise_bias=0.0,
                     output_noise=[False, False, True],
                     )

robo_ant = Task(env_name='RoboschoolAnt-v1',
                input_size=28,
                output_size=8,
                layers=[140, 40],
                time_factor=1000,
                activation=['tanh', 'tanh', 'passthru'],
                noise_bias=0.0,
                output_noise=[False, False, True],
                )

robo_walker = Task(env_name='RoboschoolWalker2d-v1',
                   input_size=22,
                   output_size=6,
                   time_factor=1000,
                   layers=[110, 30],
                   activation=['tanh', 'tanh', 'passthru'],
                   noise_bias=0.0,
                   output_noise=[False, False, True],
                   )

robo_hopper = Task(env_name='RoboschoolHopper-v1',
                   input_size=15,
                   output_size=3,
                   layers=[75, 15],
                   time_factor=1000,
                   activation=['tanh', 'tanh', 'passthru'],
                   noise_bias=0.0,
                   output_noise=[False, False, True],
                   )


tasks = {}  # task lib
tasks['bullet_pendulum'] = bullet_pendulum
tasks['bullet_double_pendulum'] = bullet_double_pendulum
tasks['bullet_minitaur_duck'] = bullet_minitaur_duck
tasks['bullet_kuka_grasping'] = bullet_kuka_grasping
tasks['bullet_kuka_grasping_stoc'] = bullet_kuka_grasping_stoc
tasks['bullet_minitaur_duck_stoc'] = bullet_minitaur_duck_stoc
tasks['bullet_minitaur_ball'] = bullet_minitaur_ball
tasks['bullet_minitaur_ball_stoc'] = bullet_minitaur_ball_stoc
tasks['bullet_half_cheetah'] = bullet_half_cheetah
tasks['bullet_humanoid'] = bullet_humanoid
tasks['bullet_ant'] = bullet_ant
tasks['bullet_walker'] = bullet_walker
tasks['bullet_hopper'] = bullet_hopper
tasks['bullet_racecar'] = bullet_racecar
tasks['bullet_minitaur'] = bullet_minitaur
tasks['bullet_minitaur_stoc'] = bullet_minitaur_stoc
tasks['bipedhard_stoc'] = bipedhard_stoc
tasks['bipedhard'] = bipedhard
tasks['carracing'] = carracing
tasks['robo_reacher'] = robo_reacher
tasks['robo_flagrun'] = robo_flagrun
tasks['robo_pendulum'] = robo_pendulum
tasks['robo_double_pendulum'] = robo_double_pendulum
tasks['robo_humanoid'] = robo_humanoid
tasks['robo_ant'] = robo_ant
tasks['robo_walker'] = robo_walker
tasks['robo_hopper'] = robo_hopper


import gym
import pybullet_envs.bullet.racecarGymEnv as racecarGymEnv
import pybullet_envs.bullet.minitaur_gym_env as minitaur_gym_env
from custom_envs.minitaur_duck import MinitaurDuckBulletEnv
from custom_envs.minitaur_ball import MinitaurBallBulletEnv
import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv


def make_env(env_name, seed=-1, render=False):
    if env_name.startswith('RacecarBulletEnv'):
        env = racecarGymEnv.RacecarGymEnv(renders=render, isDiscrete=False)
    elif env_name.startswith('MinitaurBulletEnv'):
        env = minitaur_gym_env.MinitaurBulletEnv(render=render)
    elif env_name.startswith('MinitaurDuckBulletEnv'):
        env = MinitaurDuckBulletEnv(render=render)
    elif env_name.startswith('MinitaurBallBulletEnv'):
        env = MinitaurBallBulletEnv(render=render)
    elif env_name.startswith('KukaBulletEnv'):
        env = kukaGymEnv.KukaGymEnv(renders=render, isDiscrete=False)
    else:
        if env_name.startswith('Roboschool'):
            import roboschool
        env = gym.make(env_name)
        if render and not env_name.startswith('Roboschool'):
            env.render('human')

    if seed >= 0:
        env.seed(seed)

    return env
