import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_env import CounterUASGymEnv

class PrintCallback(BaseCallback):
    def __init__(self, print_freq=1000):
        super().__init__()
        self.print_freq = print_freq
        self.episode_rewards = []
        self.best_avg = 12.0
    
    def _on_step(self):
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    reward = self.locals['rewards'][i]
                    self.episode_rewards.append(reward)
                    
                    if len(self.episode_rewards) % self.print_freq == 0:
                        avg = np.mean(self.episode_rewards[-self.print_freq:])
                        intercepted = info.get('intercepted', '?')
                        breaches = info.get('breaches', '?')
                        friendlies = info.get('friendlies_lost', '?')
                        print(f"Ep {len(self.episode_rewards)} | Avg: {avg:.2f} | Last: {reward:.2f} | Int: {intercepted} | Breach: {breaches} | Lost: {friendlies}")

                        if avg > self.best_avg:
                            self.best_avg = avg
                            self.model.save("ppo_counter_uas_best_v3")
                            print(f"---New Best: {avg:.2f} | Model saved to ppo_counter_uas_best.zip---")
        return True

def make_env():
    def _init():
        return CounterUASGymEnv()
    return _init

def make_env_s():
    # This ensures we return the actual environment instance
      # Make sure this matches your file/class name
    from sb3_env import CounterUASGymEnv
    return CounterUASGymEnv()

if __name__ == "__main__":
    num_envs = 20
    #envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    envs = SubprocVecEnv([make_env_s for _ in range(num_envs)])
    
    '''
    model = PPO(
    "MlpPolicy",
    envs,
    learning_rate=0.0002,
    n_steps=4096,
    batch_size=1024,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.02,
    clip_range=0.2,
    verbose=0,
    device="auto"
    )
    '''

    model = PPO.load("ppo_counter_uas_best_v2_ft", env=envs)
    #model.learning_rate = 0.00005
    model.ent_coef = 0.002
    model.learning_rate = lambda _: 0.00003
    model.clip_range = lambda _: 0.08
    #model.clip_range = 0.1

    callback = PrintCallback(print_freq=500)
    model.learn(total_timesteps=500_000_000, callback=callback)
    model.save("ppo_counter_uas_best_v2_ft")
    print("Training complete. Model saved to ppo_counter_uas_v2.zip")