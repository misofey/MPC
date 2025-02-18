#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, Vector3
from visualization_msgs.msg import Marker
from embedded_msgs.msg import (
    TrajectorySetpoints,
    ASControlsEstimations,
    ASMissionFinished,
    ASStateUpdate,
    FrontRight,
)
from .acados_controller import acados_settings
from acados_template import AcadosOcpSolver

from utilities import conversions
from mpc_python.utils.path_planning import SkidpadPlanner
from controller.utilities.steering_rate_limiter import SteeringRateLimiter
from controller.utilities.longitudinal_pid import longitudinalPID


class MPCPython(Node):
    def __init__(self):
        super().__init__("mpc_python")
        common_params = [
            ("finish_delay", 2.0),
            ("stopped_threshold", 0.1),
            ("steering_off_vel_threshold", 0.3),
            ("steering_max_vel_threshold", 6.0),
            ("steering_wheel_max_speed", 6.28),
            ("max_acc_rate", 3.0),
            ("min_speed", 1.0),
            ("max_speed", 20.0),
            ("max_steer", 1.57079632679),  # PI/2
            ("max_braking", 20.0),
            ("max_throttle", 20.0),
        ]

        # PID parameters
        pid_params = [
            ("pid.k_p_list", [1.5, 1.0, 1.0]),
            ("pid.k_d_list", [0.0, 0.0, 0.0]),
            ("pid.k_i_list", [1.0, 0.75, 0.5]),
        ]

        # Velocity parameters
        velocity_params = [
            ("velocity.speed_target", 9.5),
            ("velocity.speed_ref_end_mult", 0.8),
        ]

        mpc_params = [
            ("mpc.Tf", 1.0),
            ("mpc.Nt", 100),
            ("car_model.C_nom", 40000.0),
            ("car_model.lr", 0.673),
        ]

        # Declare all parameters
        for param, value in common_params + pid_params + velocity_params + mpc_params:
            self.declare_parameter(param, value)

        # Assign parameters to instance variables
        self.finish_delay = self.get_parameter("finish_delay").value
        self.stopped_threshold = self.get_parameter("stopped_threshold").value
        self.steering_off_vel_threshold = self.get_parameter("steering_off_vel_threshold").value
        self.steering_max_vel_threshold = self.get_parameter("steering_max_vel_threshold").value
        self.steering_wheel_max_speed = self.get_parameter("steering_wheel_max_speed").value
        self.max_acc_rate = self.get_parameter("max_acc_rate").value
        self.max_steer = self.get_parameter("max_steer").value
        self.max_braking = self.get_parameter("max_braking").value
        self.max_throttle = self.get_parameter("max_throttle").value

        # PID parameters

        self.pid_settings = {
            "k_p_list": self.get_parameter("pid.k_p_list").value,
            "k_d_list": self.get_parameter("pid.k_d_list").value,
            "k_i_list": self.get_parameter("pid.k_i_list").value,
        }

        # Velocity parameters
        # self.speed_ref = self.declare_parameter("speed_ref", 0.0).value

        # if self.speed_ref == 0.0:
        #     self.speed_ref = self.get_parameter("velocity.speed_ref").value
        self.speed_ref = self.get_parameter("velocity.speed_target").value
        self.speed_ref_end_mult = self.get_parameter("velocity.speed_ref_end_mult").value

        # MPC parameters
        self.Tf = self.get_parameter("mpc.Tf").value
        self.Nt = self.get_parameter("mpc.Nt").value
        self.mpc_model_params = {
            "lr": self.get_parameter("car_model.lr").value,
            "C_nom": self.get_parameter("car_model.C_nom").value,
        }
        self.dt = self.Tf / self.Nt

        self.get_logger().info("Parameters declared and assigned")

        self.steering_limiter = SteeringRateLimiter(
            lower_vel_limit=self.steering_off_vel_threshold,
            upper_vel_limit=self.steering_max_vel_threshold,
            max_steering_rate=self.steering_wheel_max_speed,
        )

        self.ocp = None
        self.solver = None
        self.long_control = None
        self.create_controller()

        self.planner = SkidpadPlanner(self.speed_ref, self.max_acc_rate, self.Nt, self.dt, self.speed_ref_end_mult)
        self.create_subscribers()

        self.create_publishers()

        # driving state machine stuff
        self.as_state = ASStateUpdate.UNDEFINED
        self.as_driving_counter = 0
        self.ready_to_drive = True
        self.laps = 0
        self.prev_poses = [0, 0, 0]
        self.reached_the_end = False
        self.standing_still = False

        # self.x_history = []
        # self.y_history = []
        # self.speed_history = []
        # self.throttle_history = []
        # self.speed_ref = []
        # self.last_published_speed = 0.0
        # self.steps = 0

        self.starting_position = [0, 0]
        self.got_starting_pose = False

        # keeping track of state
        self.current_position = [0, 0, 0]
        self.current_velocity = [0, 0, 0]
        self.current_steering_angle = 0
        self.current_progress = 0

        # control input buffer
        self.optimization_requested = False
        self.optimization_started = False
        self.as_controls_dt = 0.004
        self.control_buffer = np.zeros([self.Nt + 1])
        self.prediction_trajectory = np.empty((self.Nt + 1, 7))
        self.reference_trajectory = np.empty((self.Nt + 1, 7))
        self.time_passed_since_buffer_start = 0
        self.time_passed_since_optimization_start = 0
        self.optimization_node_times = np.linspace(0, self.Tf, self.Nt + 1)

        self.once = False
        self.get_logger().info("Finished skidpad MPC init")

    def create_controller(self):
        self.long_control = longitudinalPID(
            self.pid_settings, self.max_acc_rate, self.max_braking, self.max_throttle, self.speed_ref, self.speed_ref
        )

        self.ocp = acados_settings(self.Tf, self.Nt, self.mpc_model_params)
        x0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        # dimension
        self.solver = AcadosOcpSolver(self.ocp)
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        self.get_logger().info("MPC solver created")

    def create_subscribers(self):
        """hook method for subscribers"""
        self._sub_car_pose = self.create_subscription(
            PoseWithCovarianceStamped, "/se/vehicle/pose", self.pose_callback, 1
        )

        self._sub_as_controls_estimations = self.create_subscription(
            ASControlsEstimations, "/embedded/from/ASControlsEstimations", self.as_controls_callback, 1
        )

        # Initialize the as state update subscriber
        self._sub_as_state = self.create_subscription(
            ASStateUpdate, "/embedded/from/ASStateUpdate", self.as_state_update_callback, 1
        )

        self._sub_front_right = self.create_subscription(
            FrontRight, "/embedded/from/FrontRight", self.front_right_callback, 1
        )

        self._sub_optmization_request = self.create_subscription(
            Point, "/ctrl/opt_requests", self.optimization_request_callback, 1
        )
        # self._subscriber_imu_acceleration = self.create_subscription(
        #     IMUAcceleration, "/embedded/from/IMUAcceleration", self.imu_acceleration_callback, 1
        # )

    def create_publishers(self):
        """hook method for publishers"""
        self.pub_setpoints = self.create_publisher(TrajectorySetpoints, "/embedded/to/TrajectorySetpoints", 1)
        # self.publisher_errors = self.create_publisher(StateErrors, "/ppc/errors", 1)
        self.pub_as_finished = self.create_publisher(ASMissionFinished, "/embedded/to/ASMissionFinished", 1)
        # self.publisher_kve = self.create_publisher(ASControlsEstimations, "/ppc/uv_estimates", 1)
        # self.publisher_circle_center = self.create_publisher(Marker, "/ppc/prediction_center", 1)
        self.pub_prediction_trajectory = self.create_publisher(Marker, "/ppc/prediction_path", 1)
        self.pub_reference_trajectory = self.create_publisher(Marker, "/ppc/reference_path", 1)
        self.pub_optimization_request = self.create_publisher(Point, "/ctrl/opt_requests", 1)

    """callback functions"""

    def pose_callback(self, msg: PoseWithCovarianceStamped):

        if not (self.ready_to_drive and (self.as_state == ASStateUpdate.AS_DRIVING)):
            return

        _, _, yaw = conversions.euler_from_quaternion(msg.pose.pose.orientation)

        # set starting position of the car for estimated velocity profile
        if not self.got_starting_pose:
            self.starting_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            self.planner.set_starting_position(self.starting_position[0])

        # Update state
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

        [self.current_progress, lap_correct] = self.planner.request_progress(
            self.current_position[0], self.current_position[1], self.laps
        )
        # increment lapcounter
        if (np.mean(self.prev_poses) < 16) and (self.current_position[0] / 4 + np.mean(self.prev_poses) * 3 / 4 > 16):
            self.laps += 1

        self.prev_poses.pop(0)
        self.prev_poses.append(msg.pose.pose.position.x)

        # ask for optimization
        if not self.optimization_requested:
            request = Point()
            request.x = 1.0
            self.pub_optimization_request.publish(request)
            self.optimization_requested = True

    def optimization_request_callback(self, msg: Point):
        x = self.current_position[0]
        y = self.current_position[1]
        heading = self.current_position[2]
        status = self.solver.get_status()
        # self.get_logger().info(f'STATUS IS {status}')
        # self.get_logger().info(f'OPTIM REQUESTED IS {self.optimization_requested}')
        if not msg.x == 1:
            return
        if not self.optimization_started and self.optimization_requested:
            self.optimization_started = True
            self.time_passed_since_optimization_start = 0

            # plan target trajectoy
            (target_positions, estimated_speeds, self.current_progress, heading_derotation) = (
                self.planner.request_waypoints(x, y, heading, self.laps)
            )

            x0 = np.array(
                [
                    0,
                    0,
                    1,
                    0,
                    self.current_velocity[1],
                    self.current_velocity[2],
                    self.current_steering_angle * 0.4 / np.pi * 2,
                    # self.current_steering_angle * -0.4 / np.pi / 2,
                    # self.current_steering_angle * 3.3079 * 0.4 / np.pi,
                    # self.current_steering_angle * 14.2110 * 0.4 / np.pi,
                    # self.current_steering_angle * 0.4 / np.pi,
                    # self.current_velocity[2] * 3.3079 / 14.2119,
                    # self.current_velocity[2],
                    # self.current_velocity[2] / 14.2119,
                    # self.current_velocity[1],
                    # self.current_velocity[1] / 3.3079 * 14.2119,
                    # self.current_velocity[1] / 3.3079,
                ]
            )
            ubx0 = x0
            # ubx0[6] = 0.4
            lbx0 = x0
            # lbx0[6] = -0.4
            yref = np.zeros((self.Nt + 1, 6))
            yref[:, 0:4] = target_positions[:, 0:4]
            # mpc optimization
            for i in range(1, self.Nt + 1):
                self.solver.cost_set(i, "y_ref", yref[i, :])

            self.solver.set_flat("p", estimated_speeds)
            self.solver.set(0, "lbx", x0)
            self.solver.set(0, "ubx", x0)
            self.solver.set(0, "lbu", 0.5)
            self.solver.set(0, "ubu", 0.5)

            status = self.solver.solve()
            # try:
            #     u0 = self.solver.solve_for_x0(x0)
            # except:
            #     pass

            self.prediction_trajectory = self.solver.get_flat("x").reshape((-1, 7))
            self.prediction_trajectory[:, :2] = self.prediction_trajectory[:, :2] @ heading_derotation.T
            self.prediction_trajectory[:, :2] += self.current_position[:2]
            self.reference_trajectory = (target_positions[:, :2] + self.current_position[:2]) @ heading_derotation
            self.steering_rate_traj = self.solver.get_flat("u").reshape(-1, 1)

            u_traj = self.prediction_trajectory[:, 6]
            self.optimization_started = False

            # save mpc results to buffer
            self.control_buffer = u_traj
            self.time_passed_since_buffer_start = self.time_passed_since_optimization_start

            status = self.solver.get_status()
            # print("iteration time: ", self.solver.get_stats("time_tot"))
            # print("solver_result:", self.prediction_trajectory[::10, 6])
            self.optimization_requested = False
            self.once = True

    def publish_controls(self, inputs, header):
        # Update inputs
        inputs.header = header
        # inputs.acceleration = self.long_control.throttle
        # inputs.velocity_target = self.long_control.desired_speed

        self.pub_setpoints.publish(inputs)

        # Publish errors(velocity, yaw difference, crosstrack error)
        # error_msg = StateErrors()
        # error_msg.speed_error = float(self.controller.long_control.e)
        # error_msg.yaw_error = float(self.controller.yaw_error)
        # error_msg.lateral_error = float(self.controller.lateral_error)
        # error_msg.steering_integral = float(self.controller.steering_integral)
        # self.publisher_errors.publish(error_msg)

        # prediction_center_message = Marker(
        #     header=header,
        #     id=0,
        #     type=Marker.SPHERE_LIST,
        #     points=[Point(x=self.controller.prediction_center[0], y=self.controller.prediction_center[1], z=0.0)],
        #     action=Marker.ADD,
        #     scale=Vector3(x=0.5, y=0.5, z=0.5),
        #     color=ColorRGBA(r=0.3, g=0.3, b=0.3, a=1.0),
        #     lifetime=Duration(seconds=0.1).to_msg(),
        # )
        # self.publisher_circle_center.publish(prediction_center_message)

        prediction_path_message = Marker(
            header=header,
            id=1,
            type=Marker.LINE_STRIP,
            points=[Point(x=p[0], y=p[1], z=0.0) for p in self.prediction_trajectory[:, :2]],
            action=Marker.ADD,
            scale=Vector3(x=0.1, y=0.1, z=0.1),
            color=ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),
            lifetime=Duration(seconds=0.1).to_msg(),
        )
        prediction_path_message.pose.orientation.w = 1.0
        self.pub_prediction_trajectory.publish(prediction_path_message)

        reference_path_message = Marker(
            header=header,
            id=1,
            type=Marker.LINE_STRIP,
            points=[Point(x=p[0], y=p[1], z=0.0) for p in self.reference_trajectory[:, :2]],
            action=Marker.ADD,
            scale=Vector3(x=0.1, y=0.1, z=0.1),
            color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            lifetime=Duration(seconds=0.1).to_msg(),
        )
        reference_path_message.pose.orientation.w = 1.0
        self.pub_reference_trajectory.publish(reference_path_message)

        # prediction_real_path_message = Marker(
        #     header=header,
        #     id=2,
        #     type=Marker.LINE_STRIP,
        #     points=[Point(x=p[0], y=p[1], z=0.0) for p in self.controller.predictions_real],
        #     action=Marker.ADD,
        #     scale=Vector3(x=0.1, y=0.1, z=0.1),
        #     color=ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0),
        #     lifetime=Duration(seconds=0.1).to_msg(),
        # )
        # prediction_real_path_message.pose.orientation.w = 1.0
        # self.publisher_real_path_prediction.publish(prediction_real_path_message)

    def as_controls_callback(self, msg: ASControlsEstimations):
        """update velocity estimations and publish them"""

        position = np.empty([2])  # dummy variable for the update values
        yaw = 0  # another dummy variable for the update values

        # TODO: look at the pid update function why it asks for position and yaw
        self.long_control.update_values(position, yaw, msg.velocity_x)
        self.current_velocity = [msg.velocity_x, 0.2429 * msg.velocity_x * msg.yaw_rate - 0.17, msg.yaw_rate]

        self.time_passed_since_buffer_start += self.as_controls_dt
        self.time_passed_since_optimization_start += self.as_controls_dt

        if not (self.ready_to_drive and (self.as_state == ASStateUpdate.AS_DRIVING)):
            return

        if self.current_velocity[0] < self.steering_off_vel_threshold:
            self.publish_AS_finished(msg.header)  # Doesn't directly publish, there are some checks before

        # speed control
        desired_speed = self.planner.request_desired_speed(self.current_progress)

        self.long_control.desired_speed = desired_speed

        # self.get_logger().info(f"{self.optimization_node_times}")
        # self.get_logger().info(f"{self.control_buffer}")

        # lateral control
        steer = (
            np.interp(self.time_passed_since_buffer_start, self.optimization_node_times, self.control_buffer)
            * np.pi
            / 2
            / 0.4
        )
        # self.get_logger().info(f"time_passed_since_buffer_start: {self.time_passed_since_buffer_start}")

        if self.current_velocity[0] < 2.0:
            steer = 0

        inputs = TrajectorySetpoints()
        inputs.steer_angle = float(self.steering_limiter.limit_steer_target(steer, self.current_velocity[0]))
        # self.get_logger().info(f'Sended steering angle -> {inputs.steer_angle}')
        inputs.acceleration = self.long_control.pid_speed_control()
        inputs.velocity_target = float(self.long_control.desired_speed)
        # self.current_steering_angle = steer
        self.publish_controls(inputs, msg.header)

    # def imu_acceleration_callback(self, msg: IMUAcceleration):
    #     xsens_rotation_matrix = np.eye(3)
    #     acc = np.array([msg.acceleration_x, msg.acceleration_y, msg.acceleration_z]) @ xsens_rotation_matrix
    #     self.controller.kve.on_imuaccel(acc[0], acc[1])

    def as_state_update_callback(self, msg):
        if msg.new_state == ASStateUpdate.AS_DRIVING:
            self.as_driving_counter += 1
            # Hard-coded 2 second startup delay after pressing the start button on the RES.
            if self.as_driving_counter >= 20:
                self.as_state = msg.new_state
                self.as_driving_counter = 21
        else:
            self.as_driving_counter = 0
            self.as_state = msg.new_state

    def front_right_callback(self, msg: FrontRight):
        # self.current_steering_angle = (msg.steering_angle - 19000) / 52000 * np.pi
        self.current_steering_angle = msg.steering_angle
        pass

    """publish functions"""

    def publish_AS_finished(self, header):
        as_finished_msg = ASMissionFinished()
        as_finished_msg.header = header
        as_finished_msg.as_finished = False
        if self.reached_the_end:
            # self.get_logger().info('''REACHED THE END''')
            if self.current_velocity < self.stopped_threshold and not self.standing_still:
                self.finished_time = header.stamp
                self.standing_still = True
            if self.standing_still:
                current_finished_time = header.stamp

                # Added this delay to make sure that we're actually finished.
                # Since the velocity that we get is estimated, so if we get reading that says we are done,
                # we aren't sure. That's why it might be a good idea to add a delay to ensure that it is done.
                if (current_finished_time.sec - self.finished_time.sec) > self.finish_delay:
                    as_finished_msg.as_finished = True
                    self.pub_as_finished.publish(as_finished_msg)

    """calculation functions"""


def main(args=None):
    rclpy.init(args=args)

    mpc = MPCPython()

    rclpy.spin(mpc)

    while True:
        mpc.optimize()

    mpc.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
