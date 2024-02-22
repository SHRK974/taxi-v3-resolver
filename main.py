import gymnasium as gym

env = gym.make("Taxi-v3", render_mode="human")

env.reset()
screen = env.render()

for _ in range(50):
    action = env.action_space.sample()
    observation = env.step(action)
    
    reward, terminated = observation[1], observation[2]
    done = terminated
    
    screen = env.render()
    
    if done:
        break

env.close()

