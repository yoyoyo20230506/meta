import gym
from meta_rl import MetaRL

def train_meta_rl():
    env = gym.make('Breakout-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = MetaRL(state_dim, action_dim)

    episodes = 500
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            loss = agent.train_step(batch_size)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

if __name__ == "__main__":
    train_meta_rl()
