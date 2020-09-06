
# class ReloadLevels(gym.Wrapper):
#     def __init__(self, env, rollout):
#         super(ReloadLevels, self).__init__(env)
#         self.rollout = rollout
#         self.episode_reward = 0
#         self.completed = True
#         self.checkpoint_buffer = deque([], maxlen=3)
#         self.step_counter = 0
#         self.reward_max = {
#             'coinrun': 10,
#             'starpilot': 64,
#             'caveflyer': 12,
#             'dodgeball': 19,
#             'fruitbot': 32.4,
#             'chaser': 13,
#             'miner': 13,
#             'jumper': 10,
#             'leaper': 10,
#             'maze': 10,
#             'bigfish': 40,
#             'heist': 10,
#             'climber': 12.6,
#             'plunder': 30,
#             'ninja': 10,
#         }

#     def reset(self):
#         obs = self.env.reset()

#         if not self.rollout:
#             if self.completed:
#                 self.checkpoint_buffer.append(
#                     self.unwrapped.env.env.env.get_state())
#                 self.checkpoint_buffer.append(
#                     self.unwrapped.env.env.env.get_state())
#             else:
#                 self.unwrapped.env.env.env.set_state(
#                     self.checkpoint_buffer[-2])
#                 # Take default step to get correct obs
#                 obs, _, _, _ = self.env.step(4)

#         return obs

#     def step(self, action):
#         state, reward, done, info = self.env.step(action)

#         self.step_counter += 1
#         self.episode_reward += reward

#         if not self.rollout:
#             if self.step_counter % 30 == 0:
#                 self.checkpoint_buffer.append(
#                     self.unwrapped.env.env.env.get_state())

#             if done:
#                 if self.episode_reward >= self.reward_max[self.env.env_name]:
#                     self.completed = True
#                 else:
#                     self.completed = False

#                 self.episode_reward = 0
#                 self.step_counter = 0

#         return state, reward, done, info
