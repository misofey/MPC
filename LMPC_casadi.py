import casadi as cs
from matplotlib import pyplot as plt
import numpy as np
import yaml

class LMPCCasAdi:
    def __init__(self):

        self.n_states = 7
        self.n_inputs = 1
        self.n_params = 1
        self.N = 100
        self.t_f = 1
        self.dt = 0.00001

        self.problem = cs.Opti()
        
        self.error_tol = 1.05

        # Parameters
        #self.x_lin_point = self.problem.parameter(self.n_states, 1)
        #self.x_lin_point = self.problem.set_value(self.x_lin_point, np.array([0, 0, 1, 0, 0, 0, 0,]))
        self.x_lin_point = np.array([0, 0, 0, 1, 0, 0, 0,])
        
        #self.u_lin_point = self.problem.parameter(self.n_inputs, 1)
        #self.u_lin_point = self.problem.set_value(self.u_lin_point, np.array([0]))
        self.u_lin_point = np.array([0])
        
        #self.p_lin_point = self.problem.parameter(self.n_params, 1)
        self.p_lin_point = np.array([10])
        
        self.x_0 = self.problem.parameter(self.n_states, 1)
        self.reference_trajectory = self.problem.parameter(4, self.N) # x, y, heading
        
        # Decision variables
        self.u = self.problem.variable(self.n_inputs, self.N)
        self.x = self.problem.variable(self.n_states, self.N)
        
        # Constants
        self.set_constants()
        
        self.set_dynamics_w_casadi()
        #x_traj = np.array([[0., 0., 0., 0., 0., 0.]]).T
        #print(x_traj.shape)
        #for i in range(self.N-1):
        #    x_next = self.predict(x_traj[:, i], [0.5, 0.2], 0.01)[0]
        #    x_traj = np.append(x_traj, x_next, axis=1)
#
        #print(x_traj.shape)
        #plt.plot(x_traj[0, :], x_traj[1, :])
        #print(x_traj[0, :], x_traj[1, :])
        #plt.show()
        
        self.set_constraints()
        self.set_cost()
        self.set_solver()
        
    def set_solver(self):
        p_opts = {"expand":True}
        s_opts = {"max_iter": 100, "tol": 10e-16}
        self.problem.solver("ipopt",p_opts,s_opts)
        #self.problem.solver('qpoases')
        
    def get_tyre_stiffness(self) -> tuple[float, float]:
        C_data_y = np.array(
            [
                1.537405752168591e04,
                2.417765976460659e04,
                3.121158998819641e04,
                3.636055041362088e04,
            ]
        )
        C_data_x = [300, 500, 700, 900]

        Cf = np.interp((9.81 * self.m / 2) * (1 - self.x_cg), C_data_x, C_data_y)
        Cr = np.interp((9.81 * self.m / 2) * self.x_cg, C_data_x, C_data_y)

        return Cf, Cr

    def set_constants(self):
        self.m = 180  # Car mass [kg]
        self.I_z = 294  # TODO: unit
        self.wbase = 1.53  # wheel base [m]
        self.x_cg = 0.57  # C.G x location [m]
        self.lf = self.x_cg * self.wbase  # Front moment arm [m]
        self.lr = (1 - self.x_cg) * self.wbase  # Rear moment arm [m]

        [self.Cf, self.Cr] = self.get_tyre_stiffness()

        self.max_steering = 0.4
        self.max_steering_rate = (
            2 * self.max_steering
        )  # one second from full left to full right

    def set_cost(self) -> None:
        
        y = self.C@self.x + self.D@self.u
        error = y[:4, :] - self.reference_trajectory
        cost = error[0, :]@error[0, :].T
        #cost += error[1, :]@error[1, :].T
        #cost += error[2, :]@error[2, :].T
        #cost += error[3, :]@error[3, :].T
        #cost += y[4, :]@y[4, :].T

        self.problem.minimize(cost)

    def set_constraints(self) -> None:
        constraints = []
        # Constrain initial state
        constraints.append(self.x[:, 0] == self.x_0)
        
        # Dynamics constraints
        for i in range(self.N-1):
            #k1 = self.A@self.x[:, i] + self.B@x_traj[:, i]
            #k2 = self.f(self.x[:, i] + self.dt / 2 * k1)
            #k3 = self.f(self.x[:, i] + self.dt / 2 * k2)
            #k4 = self.f(self.x[:, i] + self.dt * k3)
            #states_next = x_traj[:, i] + self.dt/6 * (k1 + 2.0*k2 + 2.0*k3 + k4)
            #print(states_next)
            #x_traj = np.append(x_traj, states_next, axis=1)
            constraints.append(self.x[:, i+1] == (self.A @ self.x[:, i] + self.B @ self.u[:, i])*self.dt)
            #constraints.append(self.x[:, i+1] == self.f(self.x[:, i], self.u[:, i], self.p_lin_point)*self.dt)

        # Constrain states so linearization holds
        #constraints.append(self.x[2, 2:] < self.x_lin_point[2]*self.error_tol) # longitudinal speed
        #constraints.append(self.x[3, 2:] < self.x_lin_point[3]*self.error_tol) # lateral speed
        #constraints.append(self.x[4, 2:] < self.x_lin_point[4]*self.error_tol) # sin of steering angle
        #constraints.append(self.x[5, 2:] < self.x_lin_point[5]*self.error_tol) # cos of steering angle
        #constraints.append(self.x[6, 2:] < self.x_lin_point[6]*self.error_tol) # angular acceleration

        # Constrain inputs 
        #constraints.append(self.u[0, 2:] <   self.max_steering_rate)
        #constraints.append(self.u[0, 2:] >  -self.max_steering_rate)
        #constraints.append(self.u[1, 2:] < )
        #constraints.append(self.u[1, 2:] < )
        # Contrain rate of change of inputs
        #constraints.append(self.u[0, 2:-1] - self.u[0, 3:] < )
        #constraints.append(self.u[0, 2:-1] - self.u[0, 3:] > )
        #constraints.append(self.u[1, 2:-1] - self.u[1, 3:] < )
        #constraints.append(self.u[1, 2:-1] - self.u[1, 3:] > )

        # Trigonometric constraint
        #constraints.append(self.x[2, 2:]*self.x[2, 2:] + self.x[3, 2:]*self.x[3, 2:] == 1)
        #constraints.append(self.x[0, -1] == self.reference_trajectory[0, -1])
        #constraints.append(self.x[1, -1] == self.reference_trajectory[1, -1])
#
        self.problem.subject_to(constraints)

    def set_dynamics(self) -> None:
        sin_delta = self.x_lin_point[2, :]
        cos_delta = self.x_lin_point[3, :] 
        v_tx = self.x_lin_point[4, :]
        v_ty = self.x_lin_point[5, :]
        omega = self.x_lin_point[6, :]

        self.A = cs.DM.zeros(self.n_states, self.n_states)

        self.A[0, 2] = -v_ty
        self.A[0, 3] = v_tx
        self.A[0, 4] = -sin_delta
        self.A[1, 2] = v_tx
        self.A[1, 3] = v_ty
        self.A[1, 4] = cos_delta
        self.A[2, 2] = -omega
        self.A[2, 6] = -sin_delta
        self.A[3, 3] = omega
        self.A[3, 6] = cos_delta

    def set_dynamics_w_casadi(self):
        
        x = cs.MX.sym("x", self.n_states)
        u = cs.MX.sym("u", self.n_inputs)
        p = cs.MX.sym("p", self.n_params)
        
        p_x = x[0, 0]
        p_y = x[1, 0]
        cos_heading = x[2, 0]
        sin_heading = x[3, 0]
        v_y = x[4, 0]
        v_x = 5
        r = x[5, 0]
        wheel_angle = x[6, 0]

        steering_rate = u[0, 0]

        d_p_x = v_x * cos_heading - v_y * sin_heading
        d_p_y = v_x * sin_heading + v_y * cos_heading

        d_cos_heading = -sin_heading * r
        d_sin_heading = cos_heading * r

        d_v_y = -(self.Cf + self.Cr) / (self.m * v_x) * v_y
        d_v_y += (-v_x + (self.Cr * self.lr - self.Cf * self.lf)) / (self.m * v_x) * r
        d_v_y -= self.Cf / self.m * wheel_angle

        d_r = (self.lr * self.Cr - self.lf * self.Cf) / (self.I_z * v_x) * v_y
        d_r += (
            -(self.lf * self.lf * self.Cf + self.lr * self.lr * self.Cr)
            / (self.I_z * v_x)
            * r
        )
        d_r -= self.lf * self.Cf / self.I_z * wheel_angle

        d_steering = steering_rate

        f = cs.vertcat(
            d_p_x, d_p_y, d_cos_heading, d_sin_heading, d_v_y, d_r, d_steering
        )

        #f = cs.Function("f", [x, u, p], [f])
        
        A = cs.jacobian(f, x)
        A = cs.Function("A", [x, u, p], [A])
        self.A = A(self.x_lin_point, self.u_lin_point, self.p_lin_point)
        
        B = cs.jacobian(f, u)
        B = cs.Function("B", [x, u, p], [B])
        self.B = B(self.x_lin_point, self.u_lin_point, self.p_lin_point)
        self.C = np.array([
            [1, 0, 0, 0, 0, 0, 0], # x
            [0, 1, 0, 0, 0, 0, 0], # y
            [0, 0, 1, 0, 0, 0, 0], # cos heading
            [0, 0, 0, 1, 0, 0, 0], # sin heading
            [0, 0, 0, 0, 0, 0, 0],
        ])  
        self.D = np.array([
            [0],
            [0],
            [0],
            [0],
            [1],
        ])
        self.f = cs.Function("f", [x, u, p], [f])

        
    def nonlinear_dynamics(self, t, states, inputs):
        '''
        Class method calculating the state derivatives using nonlinear equations
        :param t: Current time
        :param states: State vector
        :param inputs: Input vector
        :return: Derivative of states
        '''
        
        if type(states) == cs.MX:
            x = states[0, :]
            y = states[1, :]
            phi = states[2, :]
            v_xi = states[3, :]
            v_eta = states[4, :]
            omega = states[5, :]
            d = inputs[0, :]
            delta = inputs[1, :]
        else:
            x = states[0]
            y = states[1]
            phi = states[2]
            v_xi = states[3]
            v_eta = states[4]
            omega = states[5]
            d = inputs[0]
            delta = inputs[1]

        # slip angles
        alpha_r = cs.arctan((-v_eta + self.l_r*omega)/(v_xi+0.001))
        alpha_f = delta - cs.arctan((v_eta + self.l_f * omega)/(v_xi+0.001))

        # tire forces
        F_xi = self.C_m1*d - self.C_m2*v_xi - self.C_m3*cs.sign(v_xi)
        F_reta = self.C_r*alpha_r
        F_feta = self.C_f*alpha_f

        # nonlinear state equations
        dx = v_xi * cs.cos(phi) - v_eta * cs.sin(phi)
        dy = v_xi * cs.sin(phi) + v_eta * cs.cos(phi)
        dphi = omega

        dv_xi = 1 / self.m * (F_xi + F_xi * cs.cos(delta) - F_feta * cs.sin(delta) + self.m * v_eta * omega)
        dv_eta = 1 / self.m * (F_reta + F_xi * cs.sin(delta) + F_feta * cs.cos(delta) - self.m * v_xi * omega)
        d_omega = 1 / self.I_z * (F_feta * self.l_f * cs.cos(delta) + F_xi * self.l_f * cs.sin(delta) - F_reta * self.l_r)
        d_states = cs.vertcat(dx, dy, dphi, dv_xi, dv_eta, d_omega)
        
        return d_states
    
    def predict(self, states, inputs, dt, t=0, method='RK4'):
        ''' Class method predicting the next state of the from previous state and given input

        :param states: State vector
        :param inputs: Input vector
        :param t: Current time
        :param dt: Sampling time
        :param method: Numerical method of solving the ODEs
        :return: Predicted state vector, predicted time
        '''
        if method == 'RK4':
            k1 = self.nonlinear_dynamics(t, states, inputs)
            k2 = self.nonlinear_dynamics(t + dt / 2, states + dt / 2 * k1, inputs)
            k3 = self.nonlinear_dynamics(t + dt / 2, states + dt / 2 * k2, inputs)
            k4 = self.nonlinear_dynamics(t + dt, states + dt * k3, inputs)
            states_next = states + dt/6 * (k1 + 2.0*k2 + 2.0*k3 + k4)
            t_next = t + dt

        elif method == 'FE':
            states_next = states + dt*self.nonlinear_dynamics(t, states, inputs)
            t_next = t + dt
        return states_next, t_next


    def optimize(self):
        # Set reference trajectory
        yref = np.repeat([[1., 10., 1., 0.]], self.N, axis=0).T
        yref[0, :] = np.linspace(0, 10, self.N)
        self.problem.set_value(self.reference_trajectory, yref)
        
        # Set initial Condition
        self.problem.set_value(self.x_0, np.array([0., 0.8, np.sqrt(2)/2, np.sqrt(2)/2, 0., 0., 0.]))
        #self.problem.set_value(self.x_0, np.array([1, 0, 1, 0, 0, 0, 0]))
        
        # TODO: Set v_x

        solution = self.problem.solve()
        
        state_traj = solution.value(self.x)
        input_traj = solution.value(self.u)
        y = self.C@state_traj + self.D@np.array([input_traj])
        error = y[:4, :] - yref
        cost = error[0, :]@error[0, :].T
        cost += error[1, :]@error[1, :].T
        print(f"Error: {error[:2, :]}")
        print(f"Cost {error[1, :]@error[1, :].T}")
        
        print(state_traj[1, :])
            
        plt.scatter(state_traj[0, :], state_traj[1, :], label="Optimized")
        plt.scatter(yref[0, :], yref[1, :], label="reference")
        plt.legend()
        plt.show()
            
if __name__=="__main__":
    
    mpc = LMPCCasAdi()
    mpc.optimize()