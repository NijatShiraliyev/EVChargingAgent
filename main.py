import matplotlib.pyplot as plt
import numpy as np
from EVChargingEnv import EVChargingEnv
from agent import DQNAgent
from collections import Counter

# --- CONFIGURATION ---
EPISODES = 2000
STATE_DIM = 11
ACTION_DIM = 6

env = EVChargingEnv()
agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

# --- TRACKING METRICS ---
training_rewards = []
test_agent_costs = []
test_dumb_costs = []
action_history = []

print("--- STARTING TRAINING ---")
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    agent.policy_net.train() # Ensure we are in training mode
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        action_history.append(action)
        
        agent.buffer.push(state, action, reward, next_state, done)
        agent.train_step()
        
        state = next_state
        total_reward += reward
        
    if episode % 30 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    
    training_rewards.append(total_reward)
    
    # Print progress every 50 episodes
    if episode % 200 == 0:
        print(f"Episode {episode:3d} | Reward: {total_reward:8.2f} | Epsilon: {agent.epsilon:.2f}")

print("Training Finished!")

# ---------------------------------------------------------
# --- TESTING AND COMPARISON ---
# ---------------------------------------------------------
print("\n--- STARTING EVALUATION (TESTING) ---")

TEST_RUNS = 50
total_serviced_all_runs = 0
total_success_all_runs = 0

for run in range(TEST_RUNS):
    # 1. Setup exact seed for fairness
    current_seed = 42 + run
    np.random.seed(current_seed) # Ensure environment randomization is consistent
    state, _ = env.reset(seed=current_seed)
    
    # Save copy for Dumb Agent
    initial_socs = env.socs.copy()
    initial_deadlines = env.deadlines.copy()
    initial_price = env.price
    initial_start_idx = env.start_idx # IMPORTANT: Copy the price index too!

    # --- A. SMART AGENT ---
    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    smart_total_cost = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, reward, done, _, info = env.step(action)
        
        smart_total_cost += info["real_cost"]
        
        # Track Success (If reward is big positive, it's a success)
        # Note: With new logic, success is > 20 (approx)
        total_success_all_runs += info["successes"]
        total_serviced_all_runs += info["successes"] + info["failures"]
            
    test_agent_costs.append(smart_total_cost) # <--- FIX: Add to list!

    # --- B. DUMB AGENT ---
    # Reset to exact same state
    np.random.seed(current_seed) # Reset seed again
    env.reset(seed=current_seed)
    env.socs = initial_socs.copy()
    env.deadlines = initial_deadlines.copy()
    env.price = initial_price
    env.start_idx = initial_start_idx # Ensure prices match
    
    state = env._get_obs()
    dumb_total_cost = 0
    done = False
    
    while not done:
        # Dumb Strategy: Charge first car not full
        action = 0
        for i in range(env.n_spots):
            if env.socs[i] < 100:
                action = i + 1
                break
                
        state, reward, done, _, info = env.step(action)
        dumb_total_cost += info["real_cost"] # <--- FIX: Use real_cost
            
    test_dumb_costs.append(dumb_total_cost)

# --- FINAL RESULTS ---
print(f"Average Smart Agent Cost: ${np.mean(test_agent_costs):.2f}")
print(f"Average Dumb Agent Cost:  ${np.mean(test_dumb_costs):.2f}")

if total_serviced_all_runs > 0:
    rate = (total_success_all_runs / total_serviced_all_runs) * 100
    print(f"Smart Agent Success Rate: {rate:.1f}%")

# ---------------------------------------------------------
# --- VISUALIZATION ---
# ---------------------------------------------------------
# Plot 1: Training Learning Curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Calculate a moving average to smooth the noisy RL curve
window_size = 20
smoothed_rewards = np.convolve(training_rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(smoothed_rewards, color='blue')
plt.title('DQN Learning Curve (Moving Average)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)

# Plot 2: Cost Comparison Bar Chart
plt.subplot(1, 2, 2)
labels = ['Smart Agent (DQN)', 'Dumb Agent (Greedy)']
means = [np.mean(test_agent_costs), np.mean(test_dumb_costs)]
plt.bar(labels, means, color=['green', 'red'])
plt.title('Average Energy Cost per Day')
plt.ylabel('Cost ($)')
for i, v in enumerate(means):
    plt.text(i, v + 0.5, f"${v:.2f}", ha='center')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# --- VISUALIZATION: ACTION vs PRICE PROFILE ---
# ---------------------------------------------------------
# Let's run ONE final episode and record everything step-by-step
state, _ = env.reset(seed=42)
prices = []
actions = []
socs_over_time = []

done = False
while not done:
    prices.append(env.price)
    socs_over_time.append(env.socs.copy()) # Save snapshot of batteries
    
    action = agent.select_action(state)
    actions.append(action)
    
    state, _, done, _, _ = env.step(action)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 1: Electricity Price (The "Signal")
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Electricity Price ($/kWh)', color='blue')
ax1.plot(prices, color='blue', linestyle='--', label='Grid Price')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot 2: Actions (When did we charge?)
ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Action Taken (Car Index)', color='red')

# We plot actions as dots. 0 = Wait, 1 = Car 1, etc.
ax2.scatter(range(len(actions)), actions, color='red', s=50, label='Charging Action')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(-0.5, 6) # 0 to 5

plt.title('Smart Agent Strategy: Charging vs. Price')
fig.tight_layout()
plt.show()