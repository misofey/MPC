# from Cython.Shadow import optimize
from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver
import casadi as cs
import types
import numpy as np
import matplotlib.pyplot as plt

try:
    from utils.modelling_tools import plot_results, plot_directions
except:
    from mpc_python.utils.modelling_tools import plot_results, plot_directions
"""
assumed:
    tyre stiffness is constant across all 4 wheels
"""
# rn assume car xcg is halfway between the axles
# TODO: get cg x location
global once
once = False


def explicit_derivative(x: [], u: [], p: [], global_params):
    global once
    m = 180
    Iz = 294
    wheelbase = 1.53
    # lr = global_params["lr"]
    # lf = wheelbase - lr

    C_data_y = np.array([1.537405752168591e04, 2.417765976460659e04, 3.121158998819641e04, 3.636055041362088e04])
    C_data_x = [300, 500, 700, 900]

    xcg = 0.57
    lf = xcg * wheelbase
    lr = (1 - xcg) * wheelbase

    C = [
        np.interp((9.81 * m / 2) * (lr / wheelbase), C_data_x, C_data_y),
        np.interp((9.81 * m / 2) * (lf / wheelbase), C_data_x, C_data_y),
    ]
    L = [lf, lr]

    (pos_x, pos_y, cos_head, sin_head, vy, r, delta) = x
    vx = p[0]
    ddelta = u[0]

    pos_x_dot_expr = vx * cos_head - vy * sin_head
    pos_y_dot_expr = vx * sin_head + vy * cos_head
    cos_head_dot_expr = -sin_head * r
    sin_head_dot_expr = cos_head * r

    A11 = -(C[0] + C[1]) / (m * (vx))
    A12 = -vx + (C[1] * L[1] - C[0] * L[0]) / (m * (vx))
    A21 = (C[1] * L[1] - C[0] * L[0]) / (Iz * (vx))
    A22 = -(L[0] * L[0] * C[0] + L[1] * L[1] * C[1]) / (Iz * (vx))

    B1 = -C[0] / m
    B2 = -(L[0] * C[0]) / Iz
    vy_dot_expr = A11 * vy + A12 * r + B1 * delta
    r_dot_expr = A21 * vy + A22 * r + B2 * delta
    delta_dot_expr = ddelta
    # if not once:
    # print("linear steering system: [", -(C[0] + C[1]) / (m * vx), ", ", vx - (C[0] * L[0] - C[1] * L[1]) / (m * vx))
    # print(
    #     "                         ",
    #     (C[0] * L[0] - C[1] * L[1]) / Iz,
    #     ", ",
    #     -(L[0] * L[0] * C[0] + L[1] * L[1] * C[1]) / (Iz * (vx + 0.1)),
    #     "]",
    # )
    # once = True
    # A = np.array([[A11, A12], [A21, A22]])
    # system_properties(A)

    return [
        pos_x_dot_expr,
        pos_y_dot_expr,
        cos_head_dot_expr,
        sin_head_dot_expr,
        vy_dot_expr,
        r_dot_expr,
        delta_dot_expr,
    ]


def create_model(global_params):
    """parameters: m, iz, tyre_stiffness"""

    model = types.SimpleNamespace()
    # constraints = types.SimpleNamespace()

    name = "RaceCar"

    # variable parameters
    vx = cs.MX.sym("vx")
    p = cs.vertcat(vx)

    # states
    pos_x = cs.MX.sym("pos_x")
    pos_y = cs.MX.sym("pos_y")
    cos_head = cs.MX.sym("cos_head")
    sin_head = cs.MX.sym("sin_head")
    vy = cs.MX.sym("vy")
    r = cs.MX.sym("r")
    delta = cs.MX.sym("delta")

    x = cs.vertcat(pos_x, pos_y, cos_head, sin_head, vy, r, delta)

    # inputs
    ddelta = cs.MX.sym("ddelta")
    u = cs.vertcat(ddelta)

    # state derivatives
    pos_x_dot = cs.MX.sym("pos_x_dot")
    pos_y_dot = cs.MX.sym("pos_y_dot")
    cos_head_dot = cs.MX.sym("cos_head_dot")
    sin_head_dot = cs.MX.sym("sin_head_dot")
    vy_dot = cs.MX.sym("vy_dot")
    r_dot = cs.MX.sym("r_dot")
    delta_dot = cs.MX.sym("delta_dot")

    xdot = cs.vertcat(pos_x_dot, pos_y_dot, cos_head_dot, sin_head_dot, vy_dot, r_dot, delta_dot)

    [pos_x_dot_expr, pos_y_dot_expr, cos_head_dot_expr, sin_head_dot_expr, vy_dot_expr, r_dot_expr, delta_dot_expr] = (
        explicit_derivative([pos_x, pos_y, cos_head, sin_head, vy, r, delta], [ddelta], [vx], global_params)
    )
    f_expl = cs.vertcat(
        pos_x_dot_expr, pos_y_dot_expr, cos_head_dot_expr, sin_head_dot_expr, vy_dot_expr, r_dot_expr, delta_dot_expr
    )

    # implicit dynamics
    f_impl = xdot - f_expl

    model.x = x
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.p = p
    model.u = u
    model.name = name

    # model.con_h_expr = delta
    return model


def create_cost(Tf, N, ocp):
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = 6  # outputs are: pos_x pos_y cos_head sin_head delta ddelta
    ny_e = nx

    ocp.solver_options.N_horizon = N

    ### Output selection matrices
    # Vx = np.zeros((ny, nx))
    Vx = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # Vx[: ny - 1, : ny - 1] = np.eye(ny - 1, ny - 1)
    ocp.cost.Vx = Vx
    Vx_e = Vx
    ocp.cost.Vx_e = Vx_e

    Vu = np.zeros((ny, nu))
    Vu[ny - 1, 0] = 1
    Vu = np.array([[0], [0], [0], [0], [0], [1]])
    ocp.cost.Vu = Vu
    Vu_e = Vu
    ocp.cost.Vu_e = Vu_e

    ### Cost matrices
    # Q = np.zeros((nx, ny))
    # R = np.zeros((nu, ny))
    W = np.zeros((ny, ny))

    # Q_e = np.zeros((nx, ny))
    # R_e = np.zeros((nu, ny))
    W_e = np.zeros((ny, ny))
    w1 = np.array([1.0, 1.0, 1e-3, 1e-1, 0, 1e-5]) * 0.1
    we = np.array([1, 1, 0.7, 0.7, 0, 0]) * 0.01
    W[:ny, :ny] = np.diag(w1)
    W_e[:ny, :ny] = np.diag(we)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.W = W
    ocp.cost.W_e = W_e  # acados just adds the two parts of the cost function think

    ocp.cost.yref = np.array([0, 0, 1, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 1, 0, 0, 0])

    ocp.constraints.idxbx = np.array([6])
    ocp.constraints.lbx = np.array([-0.4])
    ocp.constraints.ubx = np.array([0.4])

    ocp.constraints.idxbu = np.array([0])  # the 0th input has the constraints, so J_bu = [1]
    ocp.constraints.lbu = np.array([-10])
    ocp.constraints.ubu = np.array([10])

    # set intial condition
    ocp.constraints.x0 = np.array([0, 0, 1, 0, 0, 0, 0])
    ocp.parameter_values = np.array([9.0])

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"

    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tol = 1e-4
    # ocp.solver_options.nlp_solver_tol_comp = 1e-2

    ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_exact_hessian = True
    ocp.solver_options.qp_solver_warm_start = 0
    return ocp


def acados_settings(Tf, N, global_params):
    ocp = AcadosOcp()
    model_ac = AcadosModel()

    model = create_model(global_params)
    model_ac.x = model.x
    model_ac.u = model.u
    model_ac.xdot = model.xdot
    model_ac.f_expl_expr = model.f_expl_expr
    # model_ac.f_impl_expr = model.f_impl_expr
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    ocp = create_cost(Tf, N, ocp)

    return ocp


# def forward_rk4_dynamics(x, u, p, dt):
#     k1 = explicit_derivative(x, u, p)
#     x_2 = [x[i] + cs.times(k1[i], dt/2) for i in range(3)]
#     k2 = explicit_derivative(x_2, u, p)
#     x_3 = [x[i] + cs.times(k2[i], dt/2) for i in range(3)]
#     k3 = explicit_derivative(x_3, u, p)
#     x_4 = [x[i] + cs.times(k3[i], dt) for i in range(3)]
#     k4 = explicit_derivative(x_4, u, p)
#     x_new = [(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i in range(3)]
#     return x_new


def simulate(N, Tf, params):
    # x = [cs.DM(0.0)]
    x0 = cs.DM(np.array([0.0, 0.0, 1.0, 0.0, 0.05 * 3.3079, 0.05 * 14.2110, 0.05]))
    x = [x0[i] for i in range(7)]
    p = [3.0]
    u = [0]
    dt = Tf / N
    x_hist = [x]
    for i in range(N):
        k1 = explicit_derivative(x, u, p, params)
        x = [x[i] + cs.times(k1[i], dt) for i in range(7)]
        x_hist.append(x)
        # print(x)
    x_hist = np.array(x_hist)
    # print(x_hist)
    plt.plot(x_hist[:, 0, 0], x_hist[:, 1, 0])
    plt.show()


def optimize(Tf, N, global_params):
    ocp = acados_settings(Tf, N, global_params)
    T = 10.00  # maximum simulation time[s]
    steer = 0.0
    x0 = np.array([0.0, 0.0, 1.0, 0.0, steer * 3.3079, steer * 14.2110, -steer])
    # x0 = np.array(
    #     [
    #         0.0,
    #         0.0,
    #         1.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #     ]
    # )
    # dimension
    Nsim = int(T * N / Tf)
    tcomp_sum = 0
    tcomp_max = 0
    solver = AcadosOcpSolver(ocp)
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)
    waypoints = np.load("utils/waypoints.npy")

    yref = np.zeros((N + 1, 6))

    for i in range(5):
        index = 0 + 3 * i
        yref[:, :4] = waypoints[index : index + N + 1, :4]
        yref[:, 0] -= waypoints[index, 0]
        yref[:, 1] -= waypoints[index, 1] + 0.2
        # print(yref)
        for i in range(1, N + 1):
            solver.cost_set(i, "y_ref", yref[i, :])
            solver.set(i, "p", np.array([9.0]))

        status = solver.solve_for_x0(x0)
        # print("acados solver ran with status: ", status)
        # print("iteration time: ", solver.get_stats("time_tot"))
        x_ref = np.array([solver.get(i, "x") for i in range(N)])
        u_ref = x_ref[:, 6]
        # print("u: ", u_ref)

    # print("x: [", ", ".join(x_ref), "]")

    # u_ref = np.array([solver.get(i, "u") for i in range(N)])

    p_ref = np.array([solver.get(i, "p") for i in range(N)])
    # print("p: [", ", ".join(p_ref), "]")

    t = np.linspace(0, Tf, N)

    A_ref = np.array([str(solver.get_from_qp_in(i, "Q")) for i in range(N)])
    # print("p: [", ", ".join(A_ref), "]")

    plot_directions(x_ref, u_ref, p_ref, t, yref)


if __name__ == "__main__":
    Tf = 0.5  # prediction horizon
    N = 75  # number of discretization steps
    global_params = {"lr": 0.697, "C_nom": 40000}
    # simulate(N, Tf, global_params)
    optimize(Tf, N, global_params)
