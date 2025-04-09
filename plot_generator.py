from continuous_dynamics import indices
from simulator import StepSimulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import yaml
from matplotlib.lines import Line2D
import pandas as pd

sns.set_style("whitegrid")  # Use a clean grid style
cmap = sns.color_palette(
    "viridis", as_cmap=True
)  # Use 'viridis' for a perceptually uniform colormap

dt = 0.01
sim_len = 300
starting_state = [
    -5.0,
    0.0,
    1.0,
    0.0,  # starting pose
    15.0,
    0.0,
    0.0,  # starting velocity
    0.0,  # starting steering angle
    0.0,  # starting steering disturbance
    10.0,  # starting side force disturbance
]

state_names = [
    "pos_x",
    "pos_y",
    "heading_cos",
    "heading_sin",
    "vx",
    "vy",
    "r",
    "steer",
    "steering_dist",
    "force_dist",
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
        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state
        )
        state, _ = sim.simulate(1000)
        del sim.MPC_controller.solver  # Ensure garbage collection

        plt.plot(
            np.linspace(0, dt * 1000, 1000),
            state[:, 1],
            label=f"N={N}",
            linewidth=2,
            color=colors[idx],
        )

    plt.xlabel("Time (s)")
    plt.ylabel("State Variable")
    plt.legend(loc="best", fontsize=10, frameon=True)
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig("plots/n_tuning_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_q_tuning():
    q_low = 1000
    q_high = 10000000
    N = 40
    # Load Q values from a YAML file
    with open("parameters.yaml", "r") as file:
        params = yaml.safe_load(file)

    plt.figure(figsize=(10, 6))  # Set figure size

    num_lines = int(np.log10(int(q_high // q_low)))  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors
    states = []
    models = []
    for idx in range(num_lines):
        Tf = dt * N
        q = q_low * 10**idx
        params["controller"]["q"] = q
        with open("parameters.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state
        )
        state, input = sim.simulate(sim_len)
        states.append(state)
        models.append(f"{sim.model}-q:{q}")
        del sim.MPC_controller.solver  # Ensure garbage collection

        plt.plot(
            np.linspace(0, dt * sim_len, sim_len),
            state[:, 1],
            label=f"q={q}",
            linewidth=2,
            color=colors[idx],
        )
        plt.plot(
            np.linspace(0, dt * sim_len, sim_len),
            input[:, -1],
            linestyle="--",
            linewidth=2,
            color=colors[idx],
        )

    # Add labels for the main plot
    plt.xlabel("Time (s)")
    plt.ylabel("State Variable")

    compute_performance_metrics(states, models)

    # Add the main legend for q values
    # Add the main legend for q values
    main_legend = plt.legend(loc="best", fontsize=10, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot

    # Add a separate legend for line styles
    if idx == num_lines - 1:  # Add this legend only once
        custom_lines = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
        ]
        plt.legend(
            custom_lines,
            ["y position (solid)", "steering angle (dashed)"],
            loc="upper right",
            fontsize=10,
            frameon=True,
        )
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig("plots/q_tuning_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_r_tuning():

    q = 10  # 0.1
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
        r = r_low * 10**idx
        params["controller"]["q"] = q
        params["controller"]["r"] = r
        with open("parameters.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state
        )
        state, input = sim.simulate(sim_len)
        del sim.MPC_controller.solver  # Ensure garbage collection

        # Calculate control parameters
        y = state[:, 1]  # Assuming state[:, 1] is the output of interest
        rise_time = (
            next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
        )
        settling_time = next(
            (t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]),
            None,
        )
        settling_time = (sim_len - settling_time) * dt if settling_time else None
        overshoot = max(y) - y[-1]

        results.append(
            {
                "r": r,
                "Rise Time (s)": rise_time,
                "Settling Time (s)": settling_time,
                "Overshoot": overshoot,
            }
        )

        # Plot the results
        plt.plot(
            np.linspace(0, dt * sim_len, sim_len),
            state[:, 1],
            label=f"r={r}",
            linewidth=2,
            color=colors[idx],
        )
        plt.plot(
            np.linspace(0, dt * sim_len, sim_len),
            input[:, -1],
            linestyle="--",
            linewidth=2,
            color=colors[idx],
        )

    # Add labels for the main plot
    plt.xlabel("Time (s)")
    plt.ylabel("State Variable")

    # Add the main legend for r values
    main_legend = plt.legend(loc="best", fontsize=10, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot

    # Add a separate legend for line styles
    custom_lines = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
    ]
    plt.legend(
        custom_lines,
        ["y position (solid)", "steering angle (dashed)"],
        loc="upper right",
        fontsize=10,
        frameon=True,
    )

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig("plots/r_tuning_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)


def plot_all_state_response():
    N = 50
    Tf = dt * N
    sim = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model="L"
    )
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
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True
    )

    time = np.linspace(0, dt * sim_len, sim_len)
    colors = [
        cmap(i / num_subplots) for i in range(num_subplots)
    ]  # Use the cmap variable for colors

    results = []  # To store metrics for each state

    # Plot each state on a separate subplot
    for i in range(0, num_states):
        y = state[:, i]
        axes[i].plot(time, y, label=f"State x{i+1}", linewidth=2, color=colors[i])
        axes[i].set_ylabel(state_names[i])
        axes[i].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[i].grid(True)

        # Calculate metrics
        rise_time = (
            next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
        )
        settling_time = next(
            (t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]),
            None,
        )
        settling_time = (sim_len - settling_time) * dt if settling_time else None
        overshoot = max(y) - y[-1]

        results.append(
            {
                "State": f"x{i}",
                "Rise Time (s)": rise_time,
                "Settling Time (s)": settling_time,
                "Overshoot": overshoot,
            }
        )

    # Plot the input on the last subplot
    axes[-1].plot(time, input[:, -1], label="Input", linewidth=2, color=colors[-1])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Input Amplitude")
    axes[-1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/all_state_response.png", dpi=300, bbox_inches="tight")

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)

    # Save the table to a CSV file
    df.to_csv("plots/state_metrics.csv", index=False)

    plt.show()


def plot_compare_controllers():
    N = 50
    Tf = dt * N

    models = ["NL", "L", "LPV"]
    states, inputs, time_data = [], [], []

    for model in models:
        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=0, starting_state=starting_state, model=model
        )
        state, input = sim.simulate(sim_len)
        states.append(state)
        inputs.append(input)
        time_data.append(
            np.array(sim.ocp.metrics["runtime"]) * 1000
        )  # Convert to milliseconds
        del sim.MPC_controller.solver  # Ensure garbage collection

    # Compute statistical time metrics
    compute_time_metrics(time_data)

    num_states = states[0].shape[1]
    num_subplots = num_states + 1  # Include input subplot
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True
    )

    # Plot results
    time = np.linspace(0, dt * sim_len, sim_len)
    results = []  # To store metrics for each state and model
    colors = [
        cmap(i / (num_subplots * len(models)))
        for i in range(num_subplots * len(models))
    ]  # Use the cmap variable for colors
    # Plot each state on a separate subplot
    line_styles = ["-", "--", "-."]  # Define line styles for different models
    for i in range(num_states):
        for model_idx, model in enumerate(models):
            y = states[model_idx][:, i]
            axes[i].plot(
                time,
                y,
                label=f"{model} (x{i+1})",
                linewidth=2,
                color=colors[i * len(models) + model_idx],  # Keep the original colors
                linestyle=line_styles[model_idx % len(line_styles)],
            )
        axes[i].set_ylabel(state_names[i])
        axes[i].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[i].grid(True)

        # Calculate metrics for each model
        for model_idx, model in enumerate(models):
            y = states[model_idx][:, i]
            rise_time = (
                next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
            )
            settling_time = next(
                (t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]),
                None,
            )
            settling_time = (sim_len - settling_time) * dt if settling_time else None
            overshoot = max(y) - y[-1]

            results.append(
                {
                    "Model": model,
                    "State": f"x{i+1}",
                    "Rise Time (s)": rise_time,
                    "Settling Time (s)": settling_time,
                    "Overshoot": overshoot,
                }
            )

    # Plot the input on the last subplot
    for model_idx, model in enumerate(models):
        axes[-1].plot(
            time,
            inputs[model_idx][:, -1],
            label=f"{model} (Input)",
            linewidth=2,
            color=colors[model_idx],
            linestyle=line_styles[model_idx % len(line_styles)],
        )
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Input Amplitude")
    axes[-1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/compare_controllers.png", dpi=300, bbox_inches="tight")

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)

    # Save the table to a CSV file
    df.to_csv("plots/controller_comparison_metrics.csv", index=False)

    plt.show()


def plot_ekf_convergence():
    N = 200
    Tf = dt * N
    sim = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model="none"
    )
    u = 0.1 * np.sin(np.linspace(0, Tf, N))
    u = 0.1 * np.ones(N)
    u[100:] = 0
    state, input, estimate = sim.lsim(u, N)

    num_states = state.shape[1]
    num_subplots = num_states + 1  # Include input subplot
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True
    )

    time = np.linspace(0, Tf, N)
    colors = [
        cmap(i / num_subplots) for i in range(num_subplots)
    ]  # Use the cmap variable for colors

    results = []  # To store metrics for each state

    # Plot each state on a separate subplot
    for i in range(0, num_states):
        y = state[:, i]
        axes[i].plot(
            time,
            y,
            label=f"{state_names[i]} truth",
            linewidth=2,
            color=colors[i],
        )
        axes[i].plot(
            time,
            estimate[:, i],
            label=f"{state_names[i]} estimate",
            linewidth=2,
            linestyle="--",
            color=colors[i],
        )
        axes[i].set_ylabel(state_names[i])
        axes[i].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[i].grid(True)

    # Plot the input on the last subplot
    axes[-1].plot(time, input[:, -1], label="Input", linewidth=2, color=colors[-1])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Steer rate")
    axes[-1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/all_state_response.png", dpi=300, bbox_inches="tight")

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)

    # Save the table to a CSV file
    df.to_csv("plots/state_metrics.csv", index=False)

    plt.show()


def plot_all_states_only_of():
    N = 50
    Tf = dt * N
    sim = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model="OFL"
    )
    state, input, estimate, planned_path = sim.simulate_of(sim_len)
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
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True
    )

    time = np.linspace(0, dt * sim_len, sim_len)
    colors = [
        cmap(i / num_subplots) for i in range(num_subplots)
    ]  # Use the cmap variable for colors

    results = []  # To store metrics for each state
    # Plot each state on a separate subplot
    for i in range(0, num_states):
        y = state[:, i]
        axes[i].plot(
            time,
            y,
            label=f"{state_names[i]} truth",
            linewidth=2,
            color=colors[i],
        )
        axes[i].plot(
            time,
            estimate[:, i],
            label=f"{state_names[i]} estimate",
            linewidth=2,
            linestyle="--",
            color=colors[i],
        )
        axes[i].set_ylabel(state_names[i])
        axes[i].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[i].grid(True)

        # Calculate metrics
        rise_time = (
            next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
        )
        settling_time = next(
            (t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]),
            None,
        )
        settling_time = (sim_len - settling_time) * dt if settling_time else None
        overshoot = max(y) - y[-1]

        results.append(
            {
                "State": f"x{i}",
                "Rise Time (s)": rise_time,
                "Settling Time (s)": settling_time,
                "Overshoot": overshoot,
            }
        )

    # Plot the input on the last subplot
    axes[-1].plot(time, input[:, -1], label="Input", linewidth=2, color=colors[-1])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Input Amplitude")
    axes[-1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)

    # fixes in post
    axes[4].set_ylim([5, 17])

    # plot the planned path
    axes[0].plot(time, planned_path[:, 0], label="Ref", linestyle=":")
    axes[1].plot(time, planned_path[:, 1], label="Ref", linestyle=":")

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/all_state_response.png", dpi=300, bbox_inches="tight")

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)

    # Save the table to a CSV file
    df.to_csv("plots/state_metrics.csv", index=False)

    plt.show()


def plot_of_vs_l():
    N = 50
    Tf = dt * N
    sim_of = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model="OFL"
    )
    state_of, input_of, estimate_of, planned_path = sim_of.simulate_of(sim_len)
    time_data_of = (
        np.array(sim_of.ocp.metrics["runtime"]) * 1000
    )  # Convert to milliseconds
    compute_time_metrics(time_data_of)
    del sim_of.MPC_controller.solver  # Ensure garbage collection

    sim_l = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model="L"
    )
    state_l, input_l = sim_l.simulate(sim_len)
    time_data_l = (
        np.array(sim_l.ocp.metrics["runtime"]) * 1000
    )  # Convert to milliseconds
    compute_time_metrics(time_data_l)
    del sim_l.MPC_controller.solver  # Ensure garbage collection

    num_states = state_l.shape[1]
    num_subplots = num_states + 2  # Include input subplot
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True
    )

    time = np.linspace(0, dt * sim_len, sim_len)
    colors = [
        cmap(i / (num_subplots + 6)) for i in range(num_subplots + 6)
    ]  # Use the cmap variable for colors

    results_of = []  # To store metrics for each state
    results_l = []
    for i in range(0, num_states):
        y = state_of[:, i]
        axes[i].plot(
            time,
            y,
            label=f"OF {state_names[i]}",
            linewidth=2,
            color=colors[i],
        )
        results_of.append(performance_metrics(y, i))

        y = state_l[:, i]
        axes[i].plot(
            time,
            y,
            label=f"L {state_names[i]}",
            linewidth=2,
            color=colors[i + 6],
        )
        axes[i].set_ylabel(state_names[i])
        axes[i].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[i].grid(True)
        results_l.append(performance_metrics(y, i))

    # Plot the input on the last subplot
    axes[-1].plot(time, input_of[:, -1], label="Input", linewidth=2, color=colors[-1])
    axes[-1].plot(time, input_l[:, -1], label="Input", linewidth=2, color=colors[-4])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel("Input Amplitude")
    axes[-1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)

    # fixes in post
    axes[4].set_ylim([5, 17])

    # handle disturbance separately
    axes[indices["d_f"] - 1].plot(
        time,
        state_of[:, indices["d_f"]],
        linewidth=2,
        linestyle="--",
        label="d_f truth",
        color=colors[indices["d_f"] - 1],
    )
    axes[indices["d_f"] - 1].plot(
        time,
        estimate_of[:, indices["d_f"]],
        linewidth=2,
        label="d_f estimate",
        color=colors[indices["d_f"] - 1],
    )
    axes[indices["d_f"] - 1].set_ylabel(state_names[indices["d_f"]])
    axes[indices["d_f"] - 1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[indices["d_f"] - 1].grid(True)

    # plot the planned path
    axes[0].plot(time, planned_path[:, 0], label="Ref", linestyle=":")
    axes[1].plot(time, planned_path[:, 1], label="Ref", linestyle=":")

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/of_vs_l.png", dpi=300, bbox_inches="tight")

    # Print the results as a DataFrame
    df = pd.DataFrame(results_of)
    print(df)
    df.to_csv("plots/state_metrics_of.csv", index=False)  # save df to csv file
    # Print the results as a DataFrame
    df = pd.DataFrame(results_l)
    print(df)
    df.to_csv("plots/state_metrics_l.csv", index=False)  # save df to csv file

    plt.show()


def performance_metrics(y, i):
    """calculate performance metrics for one data"""
    rise_time = next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
    settling_time = next(
        (t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]),
        None,
    )
    settling_time = (sim_len - settling_time) * dt if settling_time else None
    overshoot = max(y) - y[-1]

    return {
        "State": f"x{i}",
        "Rise Time (s)": rise_time,
        "Settling Time (s)": settling_time,
        "Overshoot": overshoot,
    }


def compute_time_metrics(time_data_list: list):
    results = []

    for idx, time_data in enumerate(time_data_list):
        # Compute statistical parameters
        mean_runtime = np.mean(time_data)
        median_runtime = np.median(time_data)
        std_runtime = np.std(time_data)
        min_runtime = np.min(time_data)
        max_runtime = np.max(time_data)
        percentile_90 = np.percentile(time_data, 90)

        # Append the results for this dataset
        results.append(
            {
                "Dataset": f"Dataset {idx + 1}",
                "Mean Runtime (ms)": mean_runtime,
                "Median Runtime (ms)": median_runtime,
                "Standard Deviation (ms)": std_runtime,
                "Minimum Runtime (ms)": min_runtime,
                "Maximum Runtime (ms)": max_runtime,
                "90th Percentile Runtime (ms)": percentile_90,
            }
        )

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Print the DataFrame
    print(df)

    return df


def compute_performance_metrics(states: list, models: list = ["unknown"]):
    results = []
    num_states = states[0].shape[1]
    for i in range(num_states):
        # Calculate metrics for each model
        for model_idx, model in enumerate(models):
            y = states[model_idx][:, i]
            rise_time = (
                next((t for t, val in enumerate(y) if val >= 0.9 * y[-1]), None) * dt
            )
            settling_time = next(
                (t for t, val in enumerate(y[::-1]) if abs(val - y[-1]) > 0.02 * y[-1]),
                None,
            )
            settling_time = (sim_len - settling_time) * dt if settling_time else None
            overshoot = max(y) - y[-1]

            results.append(
                {
                    "Model": model,
                    "State": f"x{i+1}",
                    "Rise Time (s)": rise_time,
                    "Settling Time (s)": settling_time,
                    "Overshoot": overshoot,
                }
            )
    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)


if __name__ == "__main__":

    # plot_compare_controllers()
    # plot_ekf_convergence()
    # plot_all_states_only_of()
    # plot_all_state_response()
    # plot_q_tuning()
    # plot_compare_controllers()
    plot_of_vs_l()
