import numpy as np
import numba

import matplotlib.pyplot as plt
from enum import Enum

"""skidpad dimensions are hardcoded"""
r = 9.125
center = 16.75
half_width = 1.5
lap_length = 2 * r * np.pi
end_straight_length = 15
"""
lap counter:
0 - initial straight
1-4 eight half laps
5 - final straight

accel zone: from start location
braking zone 1: form the end of the timed section
braking zone 2: from the end of the end straight, hopefully in the stop zone

braking and accelerating is assumed to happen with max_accel only

working logic:
1. get current position
2. from current position and lap count get current progress
2.5 compensation for imprecise lap increments: make sure that the progress cannon decrease in a timestep too much
2. from current progress get where the car will be on the next timestep
3. get a position and a heading vector for each progress point
"""

class Field(Enum):
    pos_x = 0,
    pos_y = 1,
    heading_number = 2,
    heading_vector_cos = 3,
    heading_vector_sin = 4,
    vx = 5,
    vy = 6,
    yawrate = 7

class StepPlanner:
    def __init__(
        self,
        target_vel: float,
        Nt: float = 10,
        dt: float = 0.2,
        ramp_length: float = 0.0,
        fields: list = [Field.pos_y, Field.heading_vector_cos, Field.vy, Field.yawrate]
    ):
        self.target_vel = target_vel  # [m/s]
        self.dt = dt
        self.Nt = Nt

        self.prev_progress = 0

        self.ramp_length = ramp_length

    def progresses2position_and_heading(self, progresses: np.ndarray) -> np.ndarray:
        """turns an array of path progresses to an array of positions"""
        length = progresses.size
        positions = np.empty((length, 4))  # [x, y, head_x, head_y]

        for i in range(length):
            progress = progresses[i]
            positions[i, 0] = progress
            positions[i, 2] = 1
            positions[i, 3] = 0
            if progress < 0:
                positions[i, 1] = 0
            elif 0 <= progress and progress < self.ramp_length:
                # initial straight
                positions[i, 1] = progress / self.ramp_length
            elif self.ramp_length <=progress:
                positions[i, 1] = 1
        return positions

    @staticmethod
    def pos2progress(x, y):
        """turns a position to a path progress"""
        progress = x
        return progress

    def request_waypoints(self, x, y, heading, Nt=-1, dt=-1):
        if Nt == -1:
            Nt = self.Nt
        if dt == -1:
            dt = self.dt
            prev_progress = self.prev_progress

        current_progress = self.pos2progress(x, y)
        progresses = np.zeros((Nt + 1,))
        waypoints = np.zeros((Nt + 1, 2))
        speeds = np.zeros((Nt + 1,))

        progresses[0] = current_progress
        waypoints[0, :] = [x, y]

        speeds[:] = self.target_vel
        progresses[1:] = progresses[0] + np.cumsum(speeds[:-1]) * dt

        waypoints = self.progresses2position_and_heading(progresses)
        self.prev_progress = current_progress

        waypoints[:, 0] -= x
        waypoints[:, 1] -= y

        heading_derotation = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        waypoints[:, 2:] = waypoints[:, 2:] @ heading_derotation
        waypoints[:, :2] = waypoints[:, :2] @ heading_derotation
        return waypoints, speeds, progresses[0], heading_derotation

def test_planning():
    Nt = 30
    tf = 0.3

    dt = tf / Nt

    planner = StepPlanner(target_vel=9.0, Nt=Nt, dt=dt)

    x1 = -1
    y1 = 0
    [target_positions, _, _, _] = planner.request_waypoints(x1, y1, -0.1)

    waypoints = target_positions
    print(f"Waypoints: {waypoints}")


    colors = np.arange(waypoints.shape[0]) + 3
    plt.scatter((waypoints[:, 0]), (waypoints[:, 1]), c=colors, cmap="Reds")
    plt.xlim([0, 50])
    plt.ylim([-25, 25])
    # plt.scatter(x1, y1)
    # np.save("waypoints", waypoints)
    print(waypoints[:, 2:])
    plt.figure()
    plt.scatter(waypoints[:, 0], waypoints[:, 1])
    plt.plot([0], [0], "or")
    plt.show()


if __name__ == "__main__":
    test_planning()
    planner = StepPlanner(2)
    print(planner.progress2speed(31.85))
    print(StepPlanner.progress2position(31.25))
