import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

# Define the neural network for decision-making
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)   # First hidden layer
        self.fc2 = nn.Linear(64, 64)           # Second hidden layer
        self.fc4 = nn.Linear(64, action_size)  # Output layer with action_size neurons

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc4(x)  # No activation here (raw Q-values)

class SumTree:
    """Binary Sum Tree data structure for Prioritized Replay"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.pointer = 0

    def add(self, priority, experience):
        """Add a new experience with a given priority."""
        idx = self.pointer + self.capacity - 1
        self.data[self.pointer] = experience
        self.update(idx, priority)
        
        self.pointer += 1
        if self.pointer >= self.capacity:  # Overwrite old experiences
            self.pointer = 0

    def update(self, idx, priority):
        """Update a node's priority and propagate the change."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, s):
        """Retrieve an experience given a priority sample `s`."""
        idx = 0
        while idx < self.capacity - 1:
            left, right = 2 * idx + 1, 2 * idx + 2
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]  # Root node holds total priority

class DQNAgent:
    def __init__(self, state_size, action_size, model_path="models/pinball_model_2.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = SumTree(capacity=10000)  # Use SumTree for PER
        self.model_path = model_path

        # PER parameters
        self.epsilon = 1.0 # 1.0 for new model

        self.alpha = 0.6  # Determines how much prioritization is used
        self.beta = 0.4  # Importance-sampling weight
        self.gamma = 0.9 # Discount factor for future rewards (0.95–0.99 is common)
        self.beta_increment_per_step = 0.001
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.05
        self.learning_rate = 0.001

        # Initialize model
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience with its priority in memory."""
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        with torch.no_grad():
            target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item() * (1 - done)
            current_q_value = self.model(state_tensor)[action].item()
            td_error = abs(target - current_q_value) + 0.01  # Add small epsilon to avoid zero priority

        self.memory.add(td_error ** self.alpha, (state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Train using Prioritized Experience Replay."""
        if self.memory.pointer < batch_size:
            return  # Not enough data yet

        batch = []
        indices = []
        priorities = []

        total_priority = self.memory.total_priority()
        segment = total_priority / batch_size

        for i in range(batch_size):
            sample = random.uniform(segment * i, segment * (i + 1))
            idx, priority, experience = self.memory.get_leaf(sample)
            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack([torch.FloatTensor(s) for s in states])
        next_states = torch.stack([torch.FloatTensor(s) for s in next_states])
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Compute targets
        with torch.no_grad():
            targets = rewards + self.gamma * torch.max(self.model(next_states), dim=1)[0] * (1 - dones)

        current_q_values = self.model(states).gather(1, torch.tensor(actions).unsqueeze(1))

        # Compute importance-sampling weights
        probabilities = np.array(priorities) / total_priority
        importance_sampling_weights = (len(self.memory.data) * probabilities) ** (-self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()  # Normalize
        
        importance_sampling_weights = torch.FloatTensor(importance_sampling_weights).unsqueeze(1)
        
        # Compute loss with importance sampling weights
        loss = (importance_sampling_weights * self.criterion(current_q_values, targets.unsqueeze(1))).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in memory
        td_errors = abs(current_q_values - targets.unsqueeze(1)).detach().numpy().flatten()
        for idx, td_error in zip(indices, td_errors):
            self.memory.update(idx, td_error ** self.alpha + 0.01)

        # Increase beta over time to reduce bias
        self.beta = min(1.0, self.beta + self.beta_increment_per_step)

        # Reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state, last_action):
        """Chooses an action based on live data and last action."""
        state_with_last_action = state + [last_action]  # Add last action to state
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore (random)
        
        state_tensor = torch.FloatTensor(state_with_last_action)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return torch.argmax(action_values).item()  # Exploit (best action)

    def save_model(self):
        """Saves the model and training parameters."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.model_path)
        print("✅ Model saved successfully!")

    def load_model(self):
        """Loads the model and training parameters if a saved model exists."""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 1.0)  # Default to 1.0 if not found
            print("Loaded existing model from disk!")
        else:
            print("No saved model found. Starting fresh.")

    def end_training(self):
        """Call this when stopping the program to save progress."""
        self.save_model()

class AIBot:
    def __init__(self, robot, tracker, score):
        self.robot = robot
        self.tracker = tracker
        self.score = score
        self.prev_score = 0
        self.width = tracker.width
        self.height = tracker.height
        self.busy = True
        self.ball_lost = False  

        # Velocity tracking
        self.prev_x = tracker.ball_x
        self.prev_y = tracker.ball_y
        self.prev_time = time.time()
        self.ball_vx = 0
        self.ball_vy = 0

        # Store previous state for delayed reward attribution
        self.prev_state = None  
        self.prev_action = None  
        self.prev_reward = 0  

        # Store the last action taken
        self.last_action = "none"

        # Initialize AI
        self.agent = DQNAgent(state_size=5, action_size=4)  # Changed from 4 to 5

        # Start bot loop
        self.bot_thread = threading.Thread(target=self.bot_loop, daemon=True)
        self.bot_thread.start()

    def update_velocity(self):
        """Calculates velocity based on ball position change over time."""
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            self.ball_vx = (self.tracker.ball_x - self.prev_x) / dt
            self.ball_vy = (self.tracker.ball_y - self.prev_y) / dt

        self.prev_x = self.tracker.ball_x
        self.prev_y = self.tracker.ball_y
        self.prev_time = current_time

    def reward_function(self):
        """Defines real-world rewards based on ball behavior and score."""

        current_score = self.score()
        score_delta = current_score - self.prev_score
        # Update previous score
        self.prev_score = current_score
        
        if self.tracker.ball_in_square:
            return 0
        
        reward = 0

        # (R1) Score increase reward
        if score_delta > 0:
            reward += score_delta

        # (P1) Not flipping punishment
        if self.last_action in ["left_up", "none"] and self.tracker.ball_x > 0.5 * self.width and self.ball_vy > 20:
            reward -= 1
        elif self.last_action in ["right_up", "none"] and self.tracker.ball_x < 0.5 * self.width and self.ball_vy > 20:
            reward -= 1

        # (R2) Flipping reward
        else:
            reward += 5

        # (R3) Flipper hit reward
        if self.ball_vy < -50:
            reward += 15

        # (P1) Ball drain penalty
        if self.tracker.ball_y > 0.887 * self.height:
            reward = -100  # Overwrites other rewards if ball is lost

        # (P2) Stuck ball penalty
        threshold = 25
        if (abs(self.ball_vx) < threshold) and (abs(self.ball_vy) < threshold):
            reward -= 5  

        print(f"Reward: {round(reward, 2)} | {self.last_action} | {round(self.agent.epsilon, 4)}      ", end='\r')
        return reward

    def make_decision(self):
        """Makes a decision based on ball position, movement, and last action."""
        
        if not self.tracker.ball_in_square:  # Prevent unnecessary flipping
            
            self.update_velocity()  # Update the ball velocity

            ball_x = self.tracker.ball_x
            ball_y = self.tracker.ball_y
            ball_vx = self.ball_vx
            ball_vy = self.ball_vy
            state = [ball_x, ball_y, ball_vx, ball_vy]

            if ball_y > 0.4 * self.height:  # Prevent unnecessary flipping

                # Apply reward to the previous state-action pair
                if self.prev_state is not None and self.prev_action is not None:
                    self.apply_reward(self.prev_state, self.prev_action, self.prev_reward)

                # Convert last action to an integer
                action_map = {"left_up": 0, "right_up": 1, "both_up": 2, "none": 3}
                last_action_index = action_map.get(self.last_action, 0)  # Default to 0 if no last action

                # Choose an action with last action as input
                action = self.agent.act(state, last_action_index)

                # Perform the chosen action
                if action == 0:
                    self.robot.flip("left", "up")
                    self.robot.flip("right", "down")
                    self.last_action = "left_up"
                elif action == 1:
                    self.robot.flip("right", "up")
                    self.robot.flip("left", "down")
                    self.last_action = "right_up"
                elif action == 2:
                    self.robot.flip("left", "up")
                    self.robot.flip("right", "up")
                    self.last_action = "both_up"
                else:
                    self.robot.flip("left", "down")
                    self.robot.flip("right", "down")
                    self.last_action = "none"

                # Store the current state, last action, and reward
                self.prev_state = state
                self.prev_action = self.last_action
                self.prev_reward = self.reward_function()  # Calculate reward

    def apply_reward(self, state, action, reward):
        """Applies the stored reward to the neural network learning process."""

        action_map = {
            "left_up": 0,
            "right_up": 1,
            "both_up": 2,
            "none": 3
        }

        # Convert action to an integer
        action_index = action_map.get(action, 3)  # Default to "none" if unknown

        # Append last action to the state
        last_action_index = action_map.get(self.last_action, 0)  # Default to 0
        state_with_last_action = list(state) + [last_action_index]

        # Ball lost = end of episode
        done = self.tracker.ball_y > 0.885 * self.height

        # Store experience with last action included
        self.agent.remember(state_with_last_action, action_index, reward, state_with_last_action, done=done)

        # Train with experience replay
        self.agent.replay(batch_size=64)

    def bot_loop(self):
        """Loop that continuously makes decisions and trains in real-time."""
        while self.busy:
            self.make_decision()
            time.sleep(0.1)  # Adjust timing for real-world play
        self.agent.end_training()  # Saves progress before stopping

    def end_thread(self):
        """Stops the bot safely."""
        self.busy = False
        self.bot_thread.join()
