"""Template training/plotting/testing script.

"""
import os
import sys
from functools import partial
import matplotlib.pyplot as plt
import pybullet as p
import os
import pickle
import sys
import torch
import json

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video

def test_policy(config):

    # Define experiment parameters
    rows = 1
    cols = 3
    experimentResults = [[[] for _ in range(cols)] for _ in range(rows)]
    controllers = ['sac']

    # disturbances
    null_disturbance = {
        'disturbances': 'null'
    }
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
                    'decary_rate': 1
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
    experimentResults = [[[] for _ in range(cols)] for _ in range(rows)]

    # Set iterations and episode counter.
    ITERATIONS = int(config.task_config['episode_len_sec']*config.task_config['ctrl_freq'])

    for n in range(cols):
        steps.append(n)
   
    for n in range(len(controllers)):
        for m in range(cols):
            # Evaluation setup.
            set_device_from_config(config)
            if config.set_test_seed:
                # seed the evaluation (both controller and env) if given
                set_seed_from_config(config)
                env_seed = config.seed
            else:
                env_seed = None

            # add disturbance
            if m == 1:
                config.task_config['disturbances'] = (white_noise_disturbance['disturbances'])
                print(config.task_config['disturbances'])
            elif m == 2:
                config.task_config['disturbances'] = (impulse_disturbance['disturbances'])
                print(config.task_config['disturbances'])
            
            # Define function to create task/env.
            env_func = partial(make, config.task, seed=env_seed, output_dir=config.output_dir, **config.task_config)
            # Create the controller/control_agent.
            control_agent = make(config.algo,
                                env_func,
                                training=False,
                                checkpoint_path=os.path.join(config.output_dir, "model_latest.pt"),
                                output_dir=config.output_dir,
                                device=config.device,
                                seed=config.seed,
                                **config.algo_config)
            control_agent.reset()
            if config.restore:
                control_agent.load(os.path.join(config.restore, "model_latest.pt"))

            # Test controller.
            results = control_agent.run(n_episodes=config.algo_config.eval_batch_size,
                                        render=config.render,
                                        verbose=config.verbose,
                                        use_adv=config.use_adv)

            # Save evalution results.
            if config.eval_output_dir is not None:
                eval_output_dir = config.eval_output_dir
            else:
                eval_output_dir = os.path.join(config.output_dir, "eval")
            os.makedirs(eval_output_dir, exist_ok=True)
            # test trajs and statistics 
            eval_path = os.path.join(eval_output_dir, config.eval_output_path)
            os.makedirs(os.path.dirname(eval_path), exist_ok=True)
            with open(eval_path, "wb") as f:
                pickle.dump(results, f)
            ep_lengths = results["ep_lengths"]
            ep_returns = results["ep_returns"]
            mse = results["mse"]
            msg = "eval_ep_length {:.2f} +/- {:.2f}\n".format(ep_lengths.mean(), ep_lengths.std())
            msg += "eval_ep_return {:.3f} +/- {:.3f}\n".format(ep_returns.mean(), ep_returns.std())
            msg += "eval_mse {:.3f} +/- {:.3f}\n".format(mse.mean(), mse.std())
            print(msg)

            # Plot the experiment.
            experimentResults[n][m].append(mse.mean())


            if "frames" in results:
                save_video(os.path.join(eval_output_dir, "video.gif"), results["frames"])
            control_agent.close()
            print("Evaluation done.")
        config.task_config.pop('disturbances')
        # plot results
        plt.figure()
        plt.plot(steps, experimentResults[n])

        plt.legend()
        plt.title('disturbances added vs mean')
        plt.xlabel('type of disturbance')
        plt.ylabel('mse mean')
        plt.savefig('./figures/'+str(controllers[n])+'_figure8.png')


def main():
    # Make config.
    fac = ConfigFactory()
    fac.add_argument("--func", type=str, default="train", help="main function to run.")
    fac.add_argument("--thread", type=int, default=0, help="number of threads to use (set by torch).")
    fac.add_argument("--render", action="store_true", help="if to render in policy test.")
    fac.add_argument("--verbose", action="store_true", help="if to print states & actions in policy test.")
    fac.add_argument("--use_adv", action="store_true", help="if to evaluate against adversary.")
    fac.add_argument("--set_test_seed", action="store_true", help="if to set seed when testing policy.")
    fac.add_argument("--eval_output_dir", type=str, help="folder path to save evaluation results.")
    fac.add_argument("--eval_output_path", type=str, default="test_results.pkl", help="file path to save evaluation results.")
    config = fac.merge()
    # System settings.
    if config.thread > 0:
        # E.g. set single thread for less context switching
        torch.set_num_threads(config.thread)

    # Execute.
    test_policy(config)

if __name__ == "__main__":
    main()


