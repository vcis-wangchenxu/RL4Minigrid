import sys, os
curr_path = os.path.dirname(os.path.abspath(__file__))
pare_path = os.path.dirname(curr_path)
sys.path.append(pare_path)

import argparse
import yaml
from pathlib import Path
import datetime
# from tqdm import tqdm

import gymnasium as gym
import minigrid

from common.utils import get_logger, save_results, save_cfgs, \
                        plot_rewards, all_seed

class MergedConfig:
    def __init__(self) -> None:
        pass

class Main(object):
    def __init__(self):
        self.cfgs = MergedConfig()

    def process_yaml_cfg(self):
        ''' load yaml config
        '''
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--yaml', default='yaml_config/DQN_Empty-5x5-v0.yaml', type=str,
                            help='the path of config file')
        args = parser.parse_args()

        if args.yaml is not None:
            with open(args.yaml) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
            
                # merge config
                for cfg_type in load_cfg:
                    if load_cfg[cfg_type] is not None:
                        for k, v in load_cfg[cfg_type].items():
                            setattr(self.cfgs, k, v)

    def print_cfgs(self, cfg):
        ''' print parameters
        '''
        cfg_dict = vars(cfg)
        self.logger.info("Hyperparameters:")
        self.logger.info(''.join(['='] * 80))
        tplt = "{:^20}\t{:^20}\t{:^20}"
        self.logger.info(tplt.format("Name", "Value", "Type"))
        for k, v in cfg_dict.items():
            print (k, v)
            if v.__class__.__name__ == 'list':
                v = str(v)
            if v is None:
                v = 'None'
            if "support" in k:
                v = str(v[0])
            self.logger.info(tplt.format(k, v, str(type(v))))
        self.logger.info(''.join(['='] * 80))

    def create_dirs(self, cfg):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        task_dir = f"{curr_path}/tasks/{cfg.mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        setattr(cfg, 'task_dir', task_dir)
        Path(cfg.task_dir).mkdir(parents=True, exist_ok=True)
        model_dir = f"{task_dir}/models"
        setattr(cfg, 'model_dir', model_dir)
        res_dir = f"{task_dir}/results"
        setattr(cfg, 'res_dir', res_dir)
        log_dir = f"{task_dir}/logs"
        setattr(cfg, 'log_dir', log_dir)
        traj_dir = f"{task_dir}/traj"
        setattr(cfg, 'traj_dir', traj_dir)

    def envs_config(self, cfg):
            ''' configure environment
            '''
            env = gym.make(cfg.env_name, render_mode=cfg.env_render)
            env = minigrid.wrappers.RGBImgPartialObsWrapper(env) 

            state_shape = env.reset()[0]['image'].shape
            action_dim  = env.action_space.n

            self.logger.info(f"state_shape: {state_shape}, action_dim: {action_dim}")  # print info
            # update to cfg paramters
            setattr(cfg, 'state_shape', state_shape)
            setattr(cfg, 'action_dim', action_dim)
            return env

    def evaluate(self, cfg, trainer, env, agent):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _, eval_ep_reward, _ = trainer.test_one_episode(env, agent, cfg)
            sum_eval_reward += eval_ep_reward
        mean_eval_reward = sum_eval_reward / cfg.eval_eps
        return mean_eval_reward

    def singlerun(self, cfg):
        env = self.envs_config(cfg) # 定义env
        agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['Agent'])
        agent = agent_mod.Agent(cfg)  # create agent
        trainer_mod = __import__(f"algos.{cfg.algo_name}.trainer", fromlist=['Trainer'])
        trainer = trainer_mod.Trainer()  # create trainer

        if cfg.load_checkpoint:
            agent.load_model(f"tasks/{cfg.load_path}/models")
        self.logger.info(f"Start {cfg.mode}ing!")
        # print(f"Start {cfg.mode}ing!")
        self.logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        # print(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = [] # record rewards for all episodes
        steps   = [] # record steps for all episodes

        if cfg.mode.lower() == 'train':
            best_ep_reward = -float('inf')
            for i_ep in range(cfg.train_eps): # 控制10个进度条
                agent, ep_reward, ep_step = trainer.train_one_episode(env, agent, cfg)
                self.logger.info(f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}, Epsilong: {agent.epsilon:.3f}")
                # print(f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}") 
                rewards.append(ep_reward)
                steps.append(ep_step)

                if (i_ep+1) % cfg.eval_per_episode == 0:
                    mean_eval_reward = self.evaluate(cfg, trainer, env, agent)
                    if mean_eval_reward > best_ep_reward:
                        self.logger.info(f"Current episode {i_ep + 1} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(cfg.model_dir)

        elif cfg.mode.lower() == 'test':
            for i_ep in rang(cfg.test_eps):
                agent, ep_reward, ep_step = trainer.test_one_episode(env, agent, cfg)
                self.logger.info(f"Episode: {i_ep + 1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
            agent.save_model(cfg.model_dir)

        self.logger.info(f"Finish {cfg.mode}ing!")
        # print(f"Finish {cfg.mode}ing!")

        res_dic = {'episodes': range(len(rewards)), 'rewards': rewards, 'steps': steps}
        save_results(res_dic, cfg.res_dir)  # save results
        save_cfgs(self.cfgs, cfg.task_dir)  # save config
        plot_rewards(rewards,
                     title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}",
                     fpath=cfg.res_dir)

    def run(self):
        self.process_yaml_cfg()  # import and process yaml config
        self.create_dirs(self.cfgs)  # create dirs
        self.logger = get_logger(self.cfgs.log_dir)  # create the logger
        self.print_cfgs(self.cfgs)  # print the configuration
        all_seed(seed=self.cfgs.seed)
        self.singlerun(self.cfgs)

if __name__ == "__main__":
    main = Main()
    main.run()