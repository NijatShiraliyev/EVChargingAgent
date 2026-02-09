import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class EVChargingEnv(gym.Env):
    def __init__(self, csv_file="Germany.csv"):
        super(EVChargingEnv, self).__init__()
        
        # --- CONFIGURATION ---
        self.n_spots = 5
        self.battery_capacity = 50.0
        self.max_steps = 24
        
        # --- ACTION SPACE ---
        self.action_space = spaces.Discrete(self.n_spots + 1) # Removed seed from here (deprecated in some versions)
        
        # Load Data
        try:
            self.prices = pd.read_csv(csv_file)["Price (EUR/MWhe)"].values / 1000.0
        except:
            # Fallback if file not found (for testing)
            print("CSV not found, using random prices")
            self.prices = np.random.uniform(0.10, 0.50, size=10000)

        # --- OBSERVATION SPACE ---
        low_bounds = np.concatenate(([-5.0], np.zeros(5), np.zeros(5))) # Price can be negative!
        high_bounds = np.concatenate(([5.0], np.full(5, 100), np.full(5, 24)))
        
        self.observation_space = spaces.Box(
            low=low_bounds, 
            high=high_bounds, 
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.socs = np.random.uniform(10, 50, size=self.n_spots).astype(np.float32)
        self.deadlines = np.random.randint(5, 20, size=self.n_spots).astype(np.float32)
        
        # Pick random start time
        max_idx = len(self.prices) - self.max_steps
        self.start_idx = np.random.randint(0, max_idx)
        
        self.price = self.prices[self.start_idx]
        self.current_step = 0
        
        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {
            "socs": self.socs,
            "deadlines": self.deadlines,
            "price": self.price,
            "current_step": self.current_step
        }

    def _get_obs(self):
        # Normalize Price (Assuming max price around 0.50)
        norm_price = self.price / 0.5 
        # Normalize SOCs (0 to 100 -> 0 to 1)
        norm_socs = self.socs / 100.0
        # Normalize Deadlines (0 to 24 -> 0 to 1)
        norm_deadlines = self.deadlines / 24.0
        
        return np.concatenate(([norm_price], norm_socs, norm_deadlines)).astype(np.float32)
    
    
    def step(self, action):
        reward = 0
        done = False
        real_cost = 0.0
        
        # --- 1. APPLY ACTION ---
        if action > 0:
            car_idx = action - 1
            if self.socs[car_idx] < 100:
                old_soc = self.socs[car_idx]
                self.socs[car_idx] = min(self.socs[car_idx] + 10.0, 100.0)
                energy_added = (self.socs[car_idx] - old_soc)
                
                cost = self.price * (energy_added / 100.0) * self.battery_capacity
                real_cost = cost
                
                # SHAPED REWARD: 
                # Penalty for cost, but higher bonus for making progress (encourages action)
                reward -= cost * 2.0 
                reward += energy_added * 0.8  

        # --- 2. UPDATE ENVIRONMENT ---
        self.deadlines -= 1
        self.current_step += 1
        self.price = self.prices[self.start_idx + self.current_step] 
        success_count = 0
        failure_count = 0
        # --- 3. UPDATED PENALTIES & BONUSES ---
        for i in range(self.n_spots):
            # Urgent Penalty: Small nudge to charge cars with deadlines < 3 hours
            if self.deadlines[i] < 3 and self.socs[i] < 80:
                reward -= 0.5 

            if self.deadlines[i] <= 0:
                if self.socs[i] >= 100:
                    success_count += 1
                else:  
                    failure_count += 1
                    
                missing_soc = 100.0 - self.socs[i]
                if missing_soc > 0:
                    # Reduced harshness to prevent "fear" behavior
                    reward -= missing_soc * 0.3 
                else:
                    reward += 20.0 # Success bonus
                
                # Reset Car
                self.socs[i] = np.random.uniform(10, 50)
                remaining_time = self.max_steps - self.current_step
                self.deadlines[i] = np.random.randint(1, max(2, remaining_time))

        if self.current_step >= self.max_steps:
            done = True
        
        return self._get_obs(), reward, done, False, {"socs": self.socs, "real_cost": real_cost, "successes": success_count, "failures": failure_count}