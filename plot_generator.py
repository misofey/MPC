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
    -10.0,  # starting side force disturbance
]

# plt.rcParams["text.usetex"] = True
state_names = [
    r"P_{{x}}",
    f"$P_{{y}}$ [m]",
    r"cos($\varphi$) [-]",
    r"sin($\varphi$) [-]",
    f"$v_{{x}}$ [m/s]",
    f"$v_{{y}}$ [m/s]",
    r"r [rad/s]",
    r"$\delta$ [rad]",
    f"$\dot{{\delta}}$ [rad/s]",
    r"d_f",
]


def plot_n_tuning(model):

    N_low = 50
    N_high = 90
    N_step = 4

    time = np.linspace(0, dt * sim_len, sim_len)
    plt.figure(figsize=(10, 6))  # Set figure size

    num_lines = (N_high - N_low) // N_step + 1  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors

    for idx, N in enumerate(range(N_low, N_high + 1, N_step)):
        Tf = dt * N
        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model
        )
        state, _, reference = sim.simulate(sim_len)
        del sim.MPC_controller.solver  # Ensure garbage collection

        plt.plot(
            time,
            state[:, 1],
            label=f"N={N}",
            linewidth=2,
            color=colors[idx],
        )

    plt.plot(time, reference[:, 1], label="Reference", linestyle=":")
    plt.legend(loc="upper left", fontsize=15, frameon=True)

    plt.xlabel("Time [s]")
    plt.ylabel(f"{state_names[1]} [m]")
    plt.legend(loc="best", fontsize=15, frameon=True)
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig(f"plots/n_tuning_plot_{model}.png", dpi=300, bbox_inches="tight")
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


def plot_q_y_tuning(model):
    q_low = 0.1
    q_high = 10000
    N = 80
    time = np.linspace(0, dt * sim_len, sim_len)
    # Load Q values from a YAML file
    with open(f"parameters_{model}.yaml", "r") as file:
        params = yaml.safe_load(file)

    plt.figure(figsize=(10, 6))  # Set figure size

    num_lines = int(np.log10(int(q_high // q_low)))  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors
    states = []
    models = []
    for idx in range(num_lines):
        Tf = dt * N
        q = q_low * 10**idx
        params["controller"]["q"] = 1
        params["controller"]["Q"][1][1] = q
        with open(f"parameters_{model}.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model
        )
        state, input, reference = sim.simulate(sim_len)
        states.append(state)
        models.append(f"{sim.model}-q:{q}")
        del sim.MPC_controller.solver  # Ensure garbage collection

        plt.plot(
            np.linspace(0, dt * sim_len, sim_len),
            state[:, 1],
            label=f"$q_{{y}}={q:.1e}$",  # Use LaTeX formatting for subscript
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
    plt.xlabel("Time [s]")
    plt.ylabel(f"{state_names[1]} [m]")

    compute_performance_metrics(states, models)

    # Add the main legend for q values
    # Add the main legend for q values
    main_legend = plt.legend(loc="best", fontsize=10, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot
    plt.plot(time, reference[:, 1], label="Reference", linestyle=":")
    # Add a separate legend for line styles
    if idx == num_lines - 1:  # Add this legend only once
        custom_lines = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
            Line2D([0], [0], color="black", linewidth=2, linestyle=":"),
        ]
        plt.legend(
            custom_lines,
            ["Position y", "Steering rate", "Reference"],
            loc="upper right",
            fontsize=15,
            frameon=True,
        )
    
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig(f"plots/q_y_tuning_plot_{model}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_beta_tuning(model):
    beta_low = 1
    beta_high = 10000
    N = 50
    time = np.linspace(0, dt * sim_len, sim_len)
    # Load Q values from a YAML file
    with open(f"parameters_{model}.yaml", "r") as file:
        params = yaml.safe_load(file)

    plt.figure(figsize=(10, 6))  # Set figure size

    num_lines = int(np.log10(int(beta_high // beta_low)))  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors
    states = []
    models = []
    for idx in range(num_lines):
        Tf = dt * N
        beta = beta_low * 10**idx
        params["controller"]["beta"] = beta
        with open(f"parameters_{model}.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model
        )
        state, input, reference = sim.simulate(sim_len)
        states.append(state)
        models.append(f"{sim.model}-beta:{beta}")
        del sim.MPC_controller.solver  # Ensure garbage collection

        plt.plot(
            np.linspace(0, dt * sim_len, sim_len),
            state[:, 1],
            label=r"$\beta=$"+f"{beta:.1e}",  # Use LaTeX formatting for subscript
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
    plt.xlabel("Time [s]")
    plt.ylabel(f"{state_names[1]}")

    compute_performance_metrics(states, models)

    # Add the main legend for q values
    # Add the main legend for q values
    main_legend = plt.legend(loc="best", fontsize=15, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot
    plt.plot(time, reference[:, 1], label="Reference", linestyle=":")
    # Add a separate legend for line styles
    if idx == num_lines - 1:  # Add this legend only once
        custom_lines = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
            Line2D([0], [0], color="black", linewidth=2, linestyle=":"),
        ]
        plt.legend(
            custom_lines,
            ["Position y", "Steering rate", "Reference"],
            loc="upper right",
            fontsize=15,
            frameon=True,
        )
    
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig(f"plots/beta_tuning_plot_{model}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_initial_condition(model):
    beta_low = 1
    beta_high = 10000
    N = 50
    time = np.linspace(0, dt * sim_len, sim_len)

    plt.figure(figsize=(2, 1))  # Set figure size

    Tf = dt * N

    sim = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model
    )

    C_inv = np.linalg.pinv(sim.MPC_controller.C)
    print(f"C_inv: {C_inv.shape}")
    x_outside = C_inv@(1.2*np.ones((C_inv.shape[1], 1)))
    print(f"x_outside: {x_outside}")
    input_constraint = sim.MPC_controller.max_steering_rate
    K = sim.MPC_controller.K
    A = sim.MPC_controller.A_stability
    B = sim.MPC_controller.B_stability
    phi =A-B@K

    initial_conditions = [np.zeros((5, 1)), x_outside]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, x0 in enumerate(initial_conditions):

        # Simulate MPC
        sim = StepSimulator(N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model)
        state, input, reference = sim.simulate(sim_len)
        del sim.MPC_controller.solver

        # Simulate saturated input LQR
        lqr_state_resuts = phi@x0
        print(lqr_state_resuts)
        lqr_input_results = []
        references = np.zeros([reference.shape[0], 5])
        # TODO: comment here the sates given in waypoints
        references[:, :3] = np.concatenate((reference[:, :2], reference[:, 3:]), axis=1)

        for j, t in enumerate(time):
            x = lqr_state_resuts[:, -1]
            u = K@(x-references[j, :])
            if np.abs(u) > input_constraint:
                u = np.sign(u)*np.array([input_constraint])
            print((A@(x-references[j, :]) + B@u).shape)
            lqr_state_resuts = np.concatenate((lqr_state_resuts, (A@(x-references[j, :]) + B@u).reshape((-1, 1))), axis=1)
            lqr_input_results.append(u)

        ax = axes[i]
        print(lqr_state_resuts.shape)
        ax.plot(time, state[:, 1], label="MPC", linewidth=2,)
        ax.plot(time, lqr_state_resuts[1, :-1], label="Saturated Input LQR", linewidth=2,)

    # Add labels for the main plot
    plt.xlabel("Time [s]")
    plt.ylabel(f"{state_names[1]}")

    
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig(f"plots/beta_tuning_plot_{model}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_r_tuning(model):

    r_low = 0.01
    r_high = 10000
    N = 80
    time = np.linspace(0, dt * sim_len, sim_len)
    # Load Q values from a YAML file
    with open(f"parameters_{model}.yaml", "r") as file:
        params = yaml.safe_load(file)

    plt.figure(figsize=(10, 6))  # Set figure size

    num_lines = int(np.log10(int(r_high // r_low)))  # Number of lines
    colors = [cmap(i / num_lines) for i in range(num_lines)]  # Generate unique colors

    results = []  # To store control parameters for each r value

    for idx in range(num_lines):
        Tf = dt * N
        r = r_low * 10**idx
        params["controller"]["r"] = r
        with open(f"parameters_{model}.yaml", "w") as file:
            yaml.safe_dump(params, file)

        sim = StepSimulator(
            N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model
        )
        state, input, reference = sim.simulate(sim_len)
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
            time,
            state[:, 1],
            label=f"$r={r:.1e}$",
            linewidth=2,
            color=colors[idx],
        )
        plt.plot(
            time,
            input[:, -1],
            linestyle="--",
            linewidth=2,
            color=colors[idx],
        )

    # Add labels for the main plot
    plt.xlabel("Time [s]")
    plt.ylabel(f"{state_names[1]} [m]")

    # Add the main legend for r values
    main_legend = plt.legend(loc="best", fontsize=10, frameon=True)
    plt.gca().add_artist(main_legend)  # Ensure the main legend stays on the plot
    plt.plot(time, reference[:, 1], label="Reference", linestyle=":")
    # Add a separate legend for line styles
    custom_lines = [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--"),
        Line2D([0], [0], color="black", linewidth=2, linestyle=":"),
    ]
    plt.legend(
        custom_lines,
        ["Position y", "Steering rate", "Reference"],
        loc="upper right",
        fontsize=10,
        frameon=True,
    )

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.savefig(f"plots/r_tuning_plot_{model}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)


def plot_all_state_response(model):
    N = 50
    Tf = dt * N
    sim = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model=model
    )
    state, input, reference = sim.simulate(sim_len)
    time_data = np.array(sim.ocp.metrics["runtime"]) * 1000  # Convert to milliseconds

    compute_time_metrics([time_data])

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
            loc="upper right", fontsize=15, frameon=True
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
        loc="upper right", fontsize=15, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)
    # Plot reference on pos_y
    axes[1].plot(time, reference[:, 1], label="Reference", linestyle=":")
    axes[1].legend(loc="upper right", fontsize=15, frameon=True)
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"plots/all_state_response_{model}.png", dpi=300, bbox_inches="tight")

    # Print the results as a DataFrame
    df = pd.DataFrame(results)
    print(df)

    # Save the table to a CSV file
    df.to_csv(f"plots/state_metrics_{model}.csv", index=False)

    plt.show()


def plot_compare_controllers():

    models = ["NL", "L", "LPV"]
    N = {"NL": 50, "L": 50, "LPV": 80}

    states, inputs, time_data = [], [], []

    for model in models:
        sim = StepSimulator(
            N=N[model], Tf=N[model]*dt, acados_print_level=0, starting_state=starting_state, model=model
        )
        state, input, reference = sim.simulate(sim_len)
        states.append(np.concatenate((state[:, 1:4], state[:, 5:]), axis=1))
        inputs.append(input)
        time_data.append(
            np.array(sim.ocp.metrics["runtime"]) * 1000
        )  # Convert to milliseconds
        del sim.MPC_controller.solver  # Ensure garbage collection

    # Compute statistical time metrics
    compute_time_metrics(time_data)

    num_states = states[0].shape[1]
    num_subplots = num_states + 1 # Include input subplot
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True
    )

    # Plot results
    results = []  # To store metrics for each state and model
    time = np.linspace(0, dt * sim_len, sim_len)
    colors = [
        cmap(i / (num_subplots * len(models)))
        for i in range(num_subplots * len(models))
    ]  # Use the cmap variable for colors
    # Plot each state on a separate subplot
    line_styles = ["-", "--", "-."]  # Define line styles for different models
    for i in range(0, num_states):
        for model_idx, model in enumerate(models):
            y = states[model_idx][:, i]
            axes[i].plot(
                time,
                y,
                label=f"{model}",
                linewidth=2,
                color=colors[i * len(models) + model_idx],  # Keep the original colors
                linestyle=line_styles[model_idx % len(line_styles)],
            )
        
        if i > 3:
            axes[i].set_ylabel(state_names[i+2])
        else:       
            axes[i].set_ylabel(state_names[i+1])
        axes[i].legend(
            loc="upper right", fontsize=15, frameon=True
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
    # Plot the input on the first subplot
    axes[0].plot(time, reference[:, 1], label="Reference", linestyle=":")
    # Plot the input on the last subplot
    for model_idx, model in enumerate(models):
        axes[-1].plot(
            time,
            inputs[model_idx][:, -1],
            label=f"{model}",
            linewidth=2,
            color=colors[model_idx],
            linestyle=line_styles[model_idx % len(line_styles)],
        )
    axes[-1].set_xlabel("Time [s]")
    axes[-1].set_ylabel(state_names[-2])
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

    plotted_fields = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    num_states = len(plotted_fields)
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
    for state_id, plot_id in zip(plotted_fields, range(len(plotted_fields))):
        y = state[:, state_id]
        axes[plot_id].plot(
            time,
            y,
            label=f"{state_names[state_id]} truth",
            linewidth=2,
            color=colors[plot_id],
        )
        axes[plot_id].plot(
            time,
            estimate[:, state_id],
            label=f"{state_names[state_id]} estimate",
            linewidth=2,
            linestyle="--",
            color=colors[plot_id],
        )
        axes[plot_id].set_ylabel(state_names[state_id])
        axes[plot_id].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[plot_id].grid(True)

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
    N = 100
    Tf = dt * N
    sim_of = StepSimulator(
        N=N, Tf=Tf, acados_print_level=-1, starting_state=starting_state, model="OFL"
    )
    initial_state_estimate = np.array(
        [
            -6.0,
            1.0,
            1.0,
            0.0,  # pose
            15.0,
            1.7,
            -1.0,  # velocity
            0.0,  # steering
            0.0,
            0.0,  # disturbances
        ]
    )
    state_of, input_of, estimate_of, planned_path = sim_of.simulate_of(
        sim_len, initial_state_estimate=initial_state_estimate
    )
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

    plotted_fields = [1, 2, 3, 5, 6, 7]
    num_states = len(plotted_fields)
    num_subplots = num_states + 2  # Include input subplot
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(6, 2 * num_subplots), sharex=True
    )

    time = np.linspace(0, dt * sim_len, sim_len)
    colors = [
        cmap(i / (num_subplots + 6)) for i in range(num_subplots + 6)
    ]  # Use the cmap variable for colors

    results_of = []  # To store metrics for each state
    results_l = []
    for state_id, plot_id in zip(plotted_fields, range(len(plotted_fields))):
        y = state_of[:, state_id]
        axes[plot_id].plot(
            time,
            y,
            label=f"OF",
            linewidth=2,
            color=colors[plot_id],
        )
        results_of.append(performance_metrics(y, state_id))

        y = state_l[:, state_id]
        axes[plot_id].plot(
            time,
            y,
            label=f"L",
            linewidth=2,
            color=colors[plot_id + 6],
        )

        if state_id == indices["vy"]:
            axes[plot_id].plot(
                time,
                estimate_of[:, state_id],
                linewidth=2,
                linestyle="--",
                label=rf"OF {state_names[state_id]} est.",
                color=colors[plot_id],
            )
        axes[plot_id].set_ylabel(state_names[state_id], labelpad=5.0)
        axes[plot_id].legend(
            loc="upper right", fontsize=10, frameon=True
        )  # Place labels in the same corner
        axes[plot_id].grid(True)
        results_l.append(performance_metrics(y, state_id))

    # Plot the input on the last subplot
    axes[-1].plot(time, input_of[:, -1], label="OF", linewidth=2, color=colors[-1])
    axes[-1].plot(time, input_l[:, -1], label="L", linewidth=2, color=colors[-4])
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_ylabel(r"$\dot{\delta}$")
    axes[-1].legend(
        loc="upper right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-1].grid(True)

    # handle disturbance separately
    axes[-2].plot(
        time,
        state_of[:, indices["d_f"]],
        linewidth=2,
        label="d_f truth",
        color=colors[num_subplots],
    )
    axes[-2].plot(
        time,
        estimate_of[:, indices["d_f"]],
        linewidth=2,
        label="d_f estimate",
        linestyle="--",
        color=colors[num_subplots],
    )
    axes[-2].set_ylabel(state_names[indices["d_f"]])
    axes[-2].legend(
        loc="lower right", fontsize=10, frameon=True
    )  # Place labels in the same corner
    axes[-2].grid(True)

    # plot the planned path
    # axes[0].plot(time, planned_path[:, 0], label="Ref", linestyle=":")
    axes[0].plot(time, planned_path[:, 1], label="Ref", linestyle=":")

    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig("plots/of_vs_l.png", bbox_inches="tight")

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
    #plot_ekf_convergence()
    # plot_all_states_only_of()
    #plot_all_state_response("L")
    # plot_q_tuning()
    # plot_n_tuning("LPV")
    # plot_q_y_tuning("LPV")
    #plot_beta_tuning("L")
    plot_initial_condition("L")
    # plot_r_tuning("LPV")
    # plot_compare_controllers()
    # plot_of_vs_l()
