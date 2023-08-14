import numpy as np
import matplotlib.pyplot as plt
# Define grid world layout
# 0: Free cell, 1: Obstacle, 2: Goal
grid_layout = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Define rewards for each cell type
rewards = {
    0: -1,  # Free cell
    1: -10,  # Obstacle (penalty)
    2: 100  # Goal
}

# Define grid world dimensions
num_rows, num_cols = grid_layout.shape

 

# Define possible actions (up, down, left, right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
act = ["U", "D", "L", "R"]
act_dic = {
        "U" : (-1, 0),
        "D" : (1, 0),
        "L" : (0, -1),
        "R" : (0, 1)
}


# Define transition probabilities for each action

transition_probabilities = {
    "U": [1, 0, 0, 0],
    "D": [0, 1, 0, 0],
    "L": [0, 0, 1, 0],
    "R": [0, 0, 0, 1]
}

# Define discount factor
gamma = 1

# Define convergence tolerance for the value iteration algorithm
tolerance = 1e-6

# Value Iteration Algorithm
def value_iteration():
    # Initialize the value function for all states with zeros
    value_function = np.zeros((num_rows, num_cols))

    while True:
        # Creating copy of the current value function to check for convergence
        prev_value_function = np.copy(value_function)

        # Perform one step of value iteration for each state
        for i in range(num_rows):
            for j in range(num_cols):
                # Skip obstacles and goal states
                if grid_layout[i, j] in [1, 2]:
                    continue

                # Calculate the Q-values for all actions
                q_values = []
                for action in act:
                    
                    q_value = 0
                    for prob, trans_prob in zip(transition_probabilities[action], actions):
                        #print("transprob:", trans_prob)
                        next_i = np.clip(i + trans_prob[0], 0, num_rows - 1)
                        next_j = np.clip(j + trans_prob[1], 0, num_cols - 1)
                        #print("prob:",prob)
                        # print("rewards:", rewards[grid_layout[next_i, next_j]] )
                        # print("value:", value_function[next_i, next_j])
                        q_value += prob * (rewards[grid_layout[next_i, next_j]] + gamma * value_function[next_i, next_j])
                    q_values.append(q_value)

                # Update the value function with the maximum Q-value
                value_function[i, j] = np.max(q_values)

        # Check for convergence
        print(value_function)
        if np.max(np.abs(value_function - prev_value_function)) < tolerance:
            break

    return value_function

# Define starting point and goal
start_position = (1, 2)  
goal_position = (13, 6)   

# Define function to backtrack the optimal path
def backtrack_path():
    # Initialize the current position as the starting point
    current_position = start_position

    # Initialize the optimal path with the starting position
    optimal_path = [current_position]

    while current_position != goal_position:
        i, j = current_position

        # Calculate the Q-values for all actions
        q_values = []
        for action in act:
            if grid_layout[i + act_dic[action][0], j + act_dic[action][1]] == 1:  # Obstacle
                q_values.append(float("-inf"))  # Mark as invalid action
                continue

            q_value = 0
            for prob, trans_prob in zip(transition_probabilities[action], actions):
                next_i = np.clip(i + trans_prob[0], 0, num_rows - 1)
                next_j = np.clip(j + trans_prob[1], 0, num_cols - 1)
                q_value += prob * (rewards[grid_layout[next_i, next_j]] + gamma * optimal_value_function[next_i, next_j])
            q_values.append(q_value)
        
        # Determine the action with the maximum Q-value
        optimal_action = actions[np.argmax(q_values)]
        # Move to the next position based on the optimal action
        next_position = (current_position[0] + optimal_action[0], current_position[1] + optimal_action[1])
        
        # Check if the next position is valid (not an obstacle)
        if grid_layout[next_position[0], next_position[1]] == 1:
            print("Optimal path not possible. Grid is surrounded by obstacles.")
            break
        
        # Update the current position
        current_position = next_position
        # Add the current position to the optimal path
        optimal_path.append(current_position)

    return optimal_path

# Run the value iteration algorithm
optimal_value_function = value_iteration()
print("Optimal Value Function:")
for i in range(num_rows):
    for j in range(num_cols):
        if grid_layout[i, j] == 2:  # Goal state
            print(" G ", end="")
        elif grid_layout[i, j] == 1:  # Obstacle
            print(" X ", end="")
        else:
            print(f" {optimal_value_function[i, j]:.2f} ", end="")
    print()

#Determine the optimal policy
optimal_policy = np.empty((num_rows, num_cols), dtype=str)
for i in range(num_rows):
    for j in range(num_cols):
        if grid_layout[i, j] in [1, 2]:  # Obstacle or goal state
            optimal_policy[i, j] = "X"
        else:
            # Calculate the Q-values for all actions
            if all(grid_layout[i + action[0], j + action[1]] == 1 for action in actions):
                optimal_policy[i, j] = "-"  # Mark as inaccessible
                continue
            q_values = []
            for action in act:
                q_value = 0
                for prob, trans_prob in zip(transition_probabilities[action], actions):
                    next_i = np.clip(i + trans_prob[0], 0, num_rows - 1)
                    next_j = np.clip(j + trans_prob[1], 0, num_cols - 1)
                    q_value += prob * (rewards[grid_layout[next_i, next_j]] + gamma * optimal_value_function[next_i, next_j])
                q_values.append(q_value)
            
            # Determine the action with the maximum Q-value
            optimal_action = actions[np.argmax(q_values)]
            
            if optimal_action == (-1, 0):
                optimal_policy[i, j] = "U"  # Up
            elif optimal_action == (1, 0):
                optimal_policy[i, j] = "D"  # Down
            elif optimal_action == (0, -1):
                optimal_policy[i, j] = "L"  # Left
            elif optimal_action == (0, 1):
                optimal_policy[i, j] = "R"  # Right

# Print the optimal policy
print("\nOptimal Policy:")
for i in range(num_rows):
    for j in range(num_cols):
        if grid_layout[i, j] == 2:  # Goal state
            print(" G ", end="")
            optimal_policy[i,j] = "G"
        elif grid_layout[i, j] == 1:  # Obstacle
            print(" X ", end="")
        else:
            print(f" {optimal_policy[i, j]} ", end="")
    print()
# Get the optimal path
optimal_path = backtrack_path()

# Print the optimal path
print("Optimal Path:")
for i in range(num_rows):
    for j in range(num_cols):
        if grid_layout[i, j] == 2:  # Goal state
            print(" G ", end="")
        elif grid_layout[i, j] == 1:  # Obstacle
            print(" X ", end="")
        elif (i, j) == start_position:  # Starting position
            print(" S ", end="")
        elif (i, j) in optimal_path:  # Path position
            print(" * ", end="")
        else:
            print("   ", end="")
    print()
print(optimal_path)

# Plot the optimal path
arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1)}
scale = 0.25
fig,ax = plt.subplots(figsize=(10, 10))
i=0
j=0
for r, row in enumerate(optimal_policy):

    for c, cell in enumerate(row):
        if(cell!="X" and cell!="G" and cell!="-" and ((i, j) in optimal_path)):
            plt.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.1)
        elif(cell=="X"):
            plt.plot(c, 5-r, 's',markersize=30, color = 'r')
        elif(cell=="G"):
            plt.plot(c, 5-r, '8',markersize=30, color = 'g')
        j = j+1
    j=0
    i = i+1
plt.show()


