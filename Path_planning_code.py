import tkinter as tk
import numpy as np
import random

class QLearningMaze:
    def __init__(self, master):
        self.master = master
        self.master.title("Q-Learning Maze with Obstacles")

        
        self.grid_size = 5
        self.grid = np.zeros((self.grid_size, self.grid_size))

       
        self.start_pos = (0, 0)
        self.exit_pos = (4, 4)

        
        
        self.canvas = tk.Canvas(self.master, width=400, height=400)
        self.canvas.pack()

       
        self.robot_pos = self.start_pos
        self.q_table = np.zeros((self.grid_size, self.grid_size, 4))  # 4 actions: up, down, left, right
        self.alpha = 0.1  # Learning rate
        self.epsilon = 0.2  # Exploration rate
        self.gamma = 0.9  # Discount factor , Helps the robot prioritize long-term rewards
        self.total_reward = 0 

        self.obstacles = set([(i, j) for i in range(self.grid_size) for j in range(self.grid_size)
                         if (i, j) not in [(0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (4, 3), (4, 4), (0, 0)]])


        self.draw_grid()

        
        self.start_button = tk.Button(self.master, text="Start Learning", command=self.start_learning)
        self.start_button.pack()
        
        self.reset_button = tk.Button(self.master, text="Reset Maze", command=self.reset_maze)
        self.reset_button.pack()

        # Sliders 
        self.alpha_slider = tk.Scale(self.master, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Alpha (Learning Rate)")
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.pack()

        self.epsilon_slider = tk.Scale(self.master, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Epsilon (Exploration Rate)")
        self.epsilon_slider.set(self.epsilon)
        self.epsilon_slider.pack()

        # Status
        self.status_label = tk.Label(self.master, text="Status: Waiting to start...")
        self.status_label.pack()

        self.reward_label = tk.Label(self.master, text="Current Reward: 0")
        self.reward_label.pack()

    def draw_grid(self):
        self.canvas.delete("all")
        cell_size = 400 / self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.canvas.create_rectangle(i * cell_size, j * cell_size, (i + 1) * cell_size, (j + 1) * cell_size, outline="black", fill="white")

        # Drawing obstacles
        for obs in self.obstacles:
            self.canvas.create_rectangle(obs[0] * cell_size, obs[1] * cell_size, (obs[0] + 1) * cell_size, (obs[1] + 1) * cell_size, fill="black")

        # Drawing start and exit positions
        self.canvas.create_oval(self.start_pos[0] * cell_size, self.start_pos[1] * cell_size, (self.start_pos[0] + 1) * cell_size, (self.start_pos[1] + 1) * cell_size, fill="blue")
        self.canvas.create_oval(self.exit_pos[0] * cell_size, self.exit_pos[1] * cell_size, (self.exit_pos[0] + 1) * cell_size, (self.exit_pos[1] + 1) * cell_size, fill="green")

    def reset_maze(self):
        self.robot_pos = self.start_pos
        self.q_table = np.zeros((self.grid_size, self.grid_size, 4))
        self.total_reward = 0 
        self.reward_label.config(text="Current Reward: 0")
        self.status_label.config(text="Status: Maze reset, start learning!")
        self.draw_grid()

    def start_learning(self):
        self.alpha = self.alpha_slider.get()
        self.epsilon = self.epsilon_slider.get()
        self.total_reward = 0 
        self.reward_label.config(text="Current Reward: 0")
        self.status_label.config(text="Status: Learning...")
        self.learn_path()

    def learn_path(self):
        self.robot_pos = self.start_pos
        steps = 0
        while self.robot_pos != self.exit_pos:
            steps += 1
            current_x, current_y = self.robot_pos
            if random.uniform(0, 1) < self.epsilon:
                # Exploration: choosing random action
                action = random.randint(0, 3)  # 0=up, 1=down, 2=left, 3=right
            else:
                # Exploitation: choosing best action
                action = np.argmax(self.q_table[current_x, current_y])

            next_state = self.take_action(action)
            reward = self.get_reward(next_state)
            next_x, next_y = next_state
            best_next_action = np.argmax(self.q_table[next_x, next_y])

            # Updating Q-value using theformula 
            self.q_table[current_x, current_y, action] += self.alpha * (reward + self.gamma * self.q_table[next_x, next_y, best_next_action] - self.q_table[current_x, current_y, action])

            self.robot_pos = next_state
            self.update_robot_position()

            # Updating total reward and showing it
            self.total_reward += reward
            self.reward_label.config(text=f"Current Reward: {reward} | Total Reward: {self.total_reward}")

            if self.robot_pos == self.exit_pos:
                self.status_label.config(text=f"Status: Exit reached in {steps} steps!")
                return

            # showing updates after some time,that is 50 steps rather than continous
            if steps % 50 == 0: 
                self.master.after(1)

    def take_action(self, action):
        x, y = self.robot_pos
        if action == 0:  #up
            return (x, y - 1) if y > 0 else (x, y)  #Outside the grid will return the same position
        elif action == 1:  # down
            return (x, y + 1) if y < self.grid_size - 1 else (x, y)
        elif action == 2:  #left
            return (x - 1, y) if x > 0 else (x, y)
        elif action == 3:  # right
            return (x + 1, y) if x < self.grid_size - 1 else (x, y)

    def get_reward(self, state):
        if state == self.exit_pos:
            return 100  # Reward for reaching the exit
        if state in self.obstacles:
            return -1  # Penalty for hitting an obstacle
        if state == self.start_pos:
            return 0  # No reward for starting
        return 1  # Reward for moving on the path 

    def update_robot_position(self):
        self.draw_grid()
        cell_size = 400 / self.grid_size
        x, y = self.robot_pos
        self.canvas.create_oval(x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size, fill="red")
        self.canvas.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = QLearningMaze(root)
    root.mainloop()
