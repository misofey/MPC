from simulator import StepSimulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import yaml
from matplotlib.lines import Line2D
import pandas as pd

sns.set_style("whitegrid")  # Use a clean grid style
cmap = sns.color_palette("viridis", as_cmap=True)  # Use 'viridis' for a perceptually uniform colormap

dt = 0.002
sim_len = 1000
starting_state = [
    -1.0, 0.0, 1.0, 0,  # starting pose
    8.0, 0.0, 0.0,  # starting veloctiy
    0.0,  # starting steering angle
]

def plot_n_tuning():

    N_low = 20
    N_high = 40
    N_step = 1
    
    plt.figure(figsize=(10, 6))  # Set figure size
    
    num_lines = (N_high - N_low) // N_step + 1  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors
    
    for idx, N in enumerate(range(N_low, N_high + 1, N_step)):
        Tf = dt * N
        sim = StepSimulator(N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state)
        state, _ = sim.simulate(1000)
        del sim.MPC_controller.solver  # Ensure garbage collection
        
        plt.plot(np.linspace(0, dt * 1000, 1000), state[:, 1], label=f"N={N}", linewidth=2, color=colors[idx])
    
    plt.xlabel("Time (s)")
    plt.ylabel("State Variable")
    plt.legend(loc="best", fontsize=10, frameon=True)
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Save the figure
    plt.savefig("plots/n_tuning_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_q_tuning():
    q_low = 0.01
    q_high = 1000
    N = 40
    # Load Q values from a YAML file
    with open("parameters.yaml", "r") as file:
        params = yaml.safe_load(file)

    plt.figure(figsize=(10, 6))  # Set figure size
    
    num_lines = int(np.log10(int(q_high // q_low)))  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors
    
    for idx in range(num_lines):
        Tf = dt * N
        q = q_low * 10 ** idx
        params["controller"]["q"] = q
        with open("parameters.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state)
        state, input = sim.simulate(sim_len)
        del sim.MPC_controller.solver  # Ensure garbage collection
        
        plt.plot(np.linspace(0, dt * sim_len, sim_len), state[:, 1], label=f"q={q}", linewidth=2, color=colors[idx])
        plt.plot(np.linspace(0, dt * sim_len, sim_len), input[:, -1], linestyle='--', linewidth=2, color=colors[idx])
        
    # Add labels for the main plot
    plt.xlabel("Time (s)")
    plt.ylabel("State Variable")
    
    # Add the main legend for q values
    # Add the main legend for q values
    main_legend = plt.legend(loc="best", fontsize=10, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot
    
    # Add a separate legend for line styles
    if idx == num_lines - 1:  # Add this legend only once
        custom_lines = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--')
        ]
        plt.legend(custom_lines, ['y position (solid)', 'steering angle (dashed)'], loc='upper right', fontsize=10, frameon=True)
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Save the figure
    plt.savefig("plots/q_tuning_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_r_tuning():

    q = 10 # 0.1
    r_low = 0.01
    r_high = 10000
    N = 40
    # Load Q values from a YAML file
    with open("parameters.yaml", "r") as file:
        params = yaml.safe_load(file)

    plt.figure(figsize=(10, 6))  # Set figure size
    
    num_lines = int(np.log10(int(r_high // r_low)))  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors

    results = []  # To store control parameters for each r value
    
    for idx in range(num_lines):
        Tf = dt * N
        r = r_low * 10 ** idx
        params["controller"]["q"] = q
        params["controller"]["r"] = r
        with open("parameters.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state)
        state, input = sim.simulate(sim_len)
        del sim.MPC_controller.solver  # Ensure garbage collection
        
        # Calculate control parameters
        y = state[:, 1]  # Assuming state[:, 1] is the output of interest
        rise_time = next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
        settling_time = next((t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]), None)
        settling_time = (sim_len - settling_time) * dt if settling_time else None
        overshoot = max(y) - y[-1]
        
        results.append({"r": r, "Rise Time (s)": rise_time, "Settling Time (s)": settling_time, "Overshoot": overshoot})
        
        # Plot the results
        plt.plot(np.linspace(0, dt * sim_len, sim_len), state[:, 1], label=f"r={r}", linewidth=2, color=colors[idx])
        plt.plot(np.linspace(0, dt * sim_len, sim_len), input[:, -1], linestyle='--', linewidth=2, color=colors[idx])
        
    # Add labels for the main plot
    plt.xlabel("Time (s)")
    plt.ylabel("State Variable")
    
    # Add the main legend for r values
    main_legend = plt.legend(loc="best", fontsize=10, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot
    
    # Add a separate legend for line styles
    custom_lines = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-'),
        Line2D([0], [0], color='black', linewidth=2, linestyle='--')
    ]
    plt.legend(custom_lines, ['y position (solid)', 'steering angle (dashed)'], loc='upper right', fontsize=10, frameon=True)
    
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)
    
    # Save the figure
    plt.savefig("plots/r_tuning_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)


def plot_all_state_response():
    N = 300
    Tf = dt * N
    sim = StepSimulator(N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state)
    state, input = sim.simulate(sim_len)
    time_data = np.array(sim.ocp.metrics["runtime"]) * 1000  # Convert to milliseconds
    
    # Compute statistical parameters
    mean_runtime = np.mean(time_data)
    median_runtime = np.median(time_data)
    std_runtime = np.std(time_data)
    min_runtime = np.min(time_data)
    max_runtime = np.max(time_data)
    percentile_90 = np.percentile(time_data, 90)

    # Print the statistical parameters
    print("Controller Runtime Metrics (in ms):")
    print(f"Mean Runtime: {mean_runtime:.6f} ms")
    print(f"Median Runtime: {median_runtime:.6f} ms")
    print(f"Standard Deviation: {std_runtime:.6f} ms")
    print(f"Minimum Runtime: {min_runtime:.6f} ms")
    print(f"Maximum Runtime: {max_runtime:.6f} ms")
    print(f"90th Percentile Runtime: {percentile_90:.6f} ms")

    del sim.MPC_controller.solver  # Ensure garbage collection

    num_states = state.shape[1]
    num_subplots = num_states + 1  # Include input subplot
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True)

    time = np.linspace(0, dt * sim_len, sim_len)
    colors = [cmap(i / num_subplots) for i in range(num_subplots)]  # Use the cmap variable for colors

    results = []  # To store metrics for each state

    # Plot each state on a separate subplot
    for i in range(0, num_states):
        y = state[:, i]
        axes[i].plot(time, y, label=f"State x{i+1}", linewidth=2, color=colors[i])
        axes[i].set_ylabel(f"x{i} Amplitude")
        axes[i].legend(loc="upper right", fontsize=10, frameon=True)  # Place labels in the same corner
        axes[i].grid(True)

        # Calculate metrics
        rise_time = next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
        settling_time = next((t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]), None)
        settling_time = (sim_len - settling_time) * dt if settling_time else None
        overshoot = max(y) - y[-1]

        results.append({"State": f"x{i}", "Rise Time (s)": rise_time, "Settling Time (s)": settling_time, "Overshoot": overshoot})

    # Plot the input on the last subplot
    axes[-1].plot(time, input[:, -1], label="Input", linewidth=2, color=colors[-1])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Input Amplitude")
    axes[-1].legend(loc="upper right", fontsize=10, frameon=True)  # Place labels in the same corner
    axes[-1].grid(True)

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/all_state_response.png", dpi=300, bbox_inches='tight')

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)

    # Save the table to a CSV file
    df.to_csv("plots/state_metrics.csv", index=False)

    plt.show()

if __name__ == "__main__":

    plot_all_state_response()