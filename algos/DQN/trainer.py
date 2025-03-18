import numpy as np

class Trainer:
    def __init__(self):
        pass

    def preprocess_state(self, state):
        return state.astype(np.float32) / 255.0

    def train_one_episode(self, env, agent, cfg):
        ep_reward = 0
        ep_step   = 0
        state, _ = env.reset(seed = cfg.seed)
        state = self.preprocess_state(state['image'].transpose((2, 0, 1)))
        done = False
        while not done:
            ep_step += 1
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = self.preprocess_state(next_state['image'].transpose((2, 0, 1)))
            done = terminated or truncated

            if reward == 0:
                reward = -0.1

            transition = (state, action, reward, next_state, done)
            # 将轨迹放到 ReplayBuffer 中
            agent.memory.push(transition)
            
            state = next_state
            ep_reward += reward

            # 当 ReplayBuffer 数量超过一定值后，才进行 Q 网络训练
            if len(agent.memory) > cfg.batch_size:
                b_s, b_a, b_r, b_ns, b_d = agent.memory.sample(cfg.batch_size)
                transition_dict = {
                    'states': np.array(b_s),
                    'actions': np.array(b_a), 
                    'next_states': np.array(b_ns),
                    'rewards': np.array(b_r),
                    'dones': np.array(b_d)
                }
                agent.update(transition_dict)
        
        return agent, ep_reward, ep_step

    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0
        ep_step   = 0
        state, _ = env.reset(seed = cfg.seed)
        state = self.preprocess_state(state['image'].transpose((2, 0, 1)))
        done = False
        while not done:
            ep_step += 1
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = self.preprocess_state(next_state['image'].transpose((2, 0, 1)))
            done = terminated or truncated

            state = next_state
            ep_reward += reward

        return agent, ep_reward, ep_step



