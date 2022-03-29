"""A quadrotor trajectory tracking example.

Notes:
    Includes and uses PID control.

Run as:

    $ python3 ura.py --overrides ./ura.yaml

"""
import time
import pybullet as p
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
import yaml
import os

def main():
    """The main function creating, running, and closing an environment.

    """
    # Define experiment parameters
    rows = 1
    cols = 3
    experimentResults = [[[] for _ in range(cols)] for _ in range(rows)]
    controllers = ['pid']
    types_disturbances = ['none', 'white_noise', 'impulse']

    # disturbances
    white_noise_disturbance =  {
        'disturbances': {
            'observation': [
                {
                    'disturbance_func': 'white_noise',
                    'std': 0.5,
                }
            ],
            # 'action': [
            #     {
            #         'disturbance_func': 'white_noise',
            #         'std': 0.1,
            #     }
            # ]
        }
    }

    impulse_disturbance =  {
        'disturbances': {
            'observation': [
                {
                    'disturbance_func': 'impulse',
                    'magnitude': 10,
                    'step_offset': 2,
                    'duration': 1,
                    'decay_rate': 1
                }
            ],
            # 'action': [
            #     {
            #         'disturbance_func': 'impulse',
            #         'magnitude': 100,
            #         'step_offset': 5,
            #         'duration': 1,
            #         'decary_rate': 0.1
            #     }
            # ]
        }
    }

    steps = []
    rewards = []
   
    # Create an environment
    CONFIG_FACTORY = ConfigFactory()               
    config = CONFIG_FACTORY.merge()
    
    # Set iterations and episode counter.
    ITERATIONS = int(config.quadrotor_config['episode_len_sec']*config.quadrotor_config['ctrl_freq'])

    for n in range(ITERATIONS):
        steps.append(n)
    
    for n in range(1):
        for m in range(cols):
            # Start a timer.
            START = time.time()

            # add disturbance
            if m == 1:
                config.quadrotor_config['disturbances'] = (white_noise_disturbance['disturbances'])
                print(config.quadrotor_config['disturbances'])
            elif m == 2:
                config.quadrotor_config['disturbances'] = (impulse_disturbance['disturbances'])
                print(config.quadrotor_config['disturbances'])
                    
            # Create controller.
            env_func = partial(make,'quadrotor', **config.quadrotor_config)
            ctrl = make('pid',env_func,)
                        
            reference_traj = ctrl.reference

            # Plot trajectory.
            for i in range(0, reference_traj.shape[0], 10):
                p.addUserDebugLine(lineFromXYZ=[reference_traj[i-10,0], 0, reference_traj[i-10,2]],
                                lineToXYZ=[reference_traj[i,0], 0, reference_traj[i,2]],
                                lineColorRGB=[1, 0, 0],
                                physicsClientId=ctrl.env.PYB_CLIENT)

            # Run the experiment.
            results = ctrl.run(iterations=ITERATIONS)
                    
            # Plot the experiment.
            for j in range(ITERATIONS):
                # Step the environment and print all returned information.
                # obs, reward, done, info, action = results['obs'][i], results['reward'][i], results['done'][i], results['info'][i], results['action'][i]
                reward = results['reward'][j]

                # record results
                experimentResults[n][m].append(1-reward[0])

            ctrl.close()            

            elapsed_sec = time.time() - START
            # print("\n{:d} iterations (@{:d}Hz) and {:d} episodes in {:.2f} seconds, i.e. {:.2f} steps/sec for a {:.2f}x speedup.\n"
            #       .format(ITERATIONS, config.quadrotor_config.ctrl_freq, 1, elapsed_sec, ITERATIONS/elapsed_sec, (ITERATIONS*(1. / config.quadrotor_config.ctrl_freq))/elapsed_sec))
        
        # plot results
        plt.figure()
        plt.plot(steps, experimentResults[n][0], label = "no disturbance")
        plt.plot(steps, experimentResults[n][1], label = "white noise")
        plt.plot(steps, experimentResults[n][2], label = "impulse")

        plt.legend()
        plt.title('cost vs iterations')
        plt.xlabel('iterations')
        plt.ylabel('cost')
        plt.savefig('./figures/'+str(controllers[n])+'_'+str(types_disturbances[m])+'.png')
        config.quadrotor_config.pop('disturbances')


if __name__ == "__main__":
    main()