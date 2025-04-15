import numpy as np
import numba

import matplotlib.pyplot as plt

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

spec = [
    ("target_vel", numba.float32),
    ("max_accel", numba.float32),
    ("dt", numba.float32),
    ("Nt", numba.int16),
    ("accel_zone_1_start", numba.float32),
    ("accel_zone_1_end", numba.float32),
    ("braking_zone_1_start", numba.float32),
    ("braking_zone_1_end", numba.float32),
    ("braking_zone_2_start", numba.float32),
    ("braking_zone_2_end", numba.float32),
    ("slowdown_speed_factor", numba.float32),
    ("prev_progress", numba.float32),
]


# @numba.experimental.jitclass(spec)
class SkidpadPlanner:
    def __init__(
        self,
        target_vel: float,
        max_accel: float = 10,
        Nt: float = 10,
        dt: float = 0.2,
        slowdown_speed_factor=0.5,
    ):
        self.target_vel = target_vel  # [m/s]
        self.max_accel = max_accel  # [m/s^2]
        self.dt = dt
        self.Nt = Nt

        accel_time = self.target_vel / self.max_accel
        self.accel_zone_1_start = 0
        self.accel_zone_1_end = (
            self.max_accel / 2 * accel_time * accel_time
        )  # s = a/2 t^2

        self.slowdown_speed_factor = slowdown_speed_factor

        decel_time = accel_time * (1 - slowdown_speed_factor)
        self.braking_zone_1_start = center + 1 + 4 * lap_length
        self.braking_zone_1_end = (
            self.braking_zone_1_start
            + decel_time * self.target_vel
            + self.max_accel / 2 * decel_time * decel_time
        )

        decel_time_2 = accel_time * slowdown_speed_factor
        self.braking_zone_2_start = center + end_straight_length + 4 * lap_length
        self.braking_zone_2_end = (
            self.braking_zone_2_start + decel_time_2 * self.max_accel
        )
        self.prev_progress = 0

    def set_starting_position(self, progress):
        accel_time = self.target_vel / self.max_accel
        self.accel_zone_1_start = progress
        self.accel_zone_1_end = (
            self.max_accel / 2 * accel_time * accel_time + progress
        )  # s = a/2 t^2

    def progress2speed(self, progress):
        speed = 0
        if progress > self.braking_zone_2_end:
            # stop zone
            speed = 0
        elif self.braking_zone_2_end > progress > self.braking_zone_2_start:
            # braking zone 2
            speed = (
                self.target_vel * self.slowdown_speed_factor
                - (progress - self.braking_zone_2_start)
                * self.target_vel
                * self.slowdown_speed_factor
            )
        elif self.braking_zone_2_start > progress > self.braking_zone_1_end:
            # slowdown zone
            speed = self.target_vel * self.slowdown_speed_factor
        elif self.braking_zone_1_end > progress > self.braking_zone_1_start:
            # braking zone 1
            speed = self.target_vel - (
                progress - self.braking_zone_1_start
            ) * self.target_vel * (1 - self.slowdown_speed_factor)
        elif self.braking_zone_1_start > progress > self.accel_zone_1_end:
            speed = self.target_vel
        elif self.accel_zone_1_end > progress > self.accel_zone_1_start:
            speed = (progress - self.accel_zone_1_start) * (self.target_vel - 0.1) + 0.1
        else:
            speed = 0.1
            # print("path progress not found on the track for velocity target calculation")
            # print(
            #     f"waypoints: {self.accel_zone_1_start}, {self.accel_zone_1_end}, {self.braking_zone_1_start}, {self.braking_zone_1_end}, {self.braking_zone_2_start}, {self.braking_zone_2_end}"
            # )
            # print(f"waypoint looked at: {progress}")

        return speed

    @staticmethod
    # @numba.njit
    # something is wrong for the last straight, probably in this function or where the progresses are calculated
    def progresses2position_and_heading(progresses: np.ndarray) -> np.ndarray:
        """turns an array of path progresses to an array of positions"""
        lap = (progresses[0] - center) // (lap_length)
        length = progresses.size
        positions = np.empty((length, 4))  # [x, y, head_x, head_y]

        for i in range(length):
            progress = progresses[i]
            if (progress - center) // (4 * lap_length) > 0:
                # final straight
                positions[i, 0] = progress - 4 * lap_length
                positions[i, 1] = 0
                positions[i, 2] = 1
                positions[i, 3] = 0
            elif ((progress - center) // (2 * lap_length)) > 0:
                # left side laps
                positions[i, 0] = (
                    center + np.sin((progress - center - 2 * lap_length) / r) * r
                )
                positions[i, 1] = (
                    r - np.cos((progress - center - 2 * lap_length) / r) * r
                )
                positions[i, 2] = np.cos((progress - center) / r)
                positions[i, 3] = np.sin((progress - center) / r)
            elif (progress - center) > 0:
                # right side laps
                positions[i, 0] = center + np.sin((progress - center) / r) * r
                positions[i, 1] = -r + np.cos((progress - center) / r) * r
                positions[i, 2] = np.cos((progress - center) / r)
                positions[i, 3] = -np.sin((progress - center) / r)
            elif progress < center:
                # initial straight
                positions[i, 0] = progress
                positions[i, 1] = 0
                positions[i, 2] = 1
                positions[i, 3] = 0
        return positions

    @staticmethod
    # @numba.njit
    def progress2position(progress: np.ndarray) -> np.ndarray:
        """turns an array of path progresses to an array of positions"""
        lap = (progress - center) // (lap_length)
        positions = np.empty((2))

        if (progress - center) // (4 * lap_length) > 0:
            # final straight
            positions[0] = progress - center - 4 * lap_length
            positions[1] = 0
        elif ((progress - center) // (2 * lap_length)) > 0:
            # left side laps
            positions[0] = center + np.sin((progress - center - 2 * lap_length) / r)
            positions[1] = r + np.cos((progress - center - 2 * lap_length) / r)
        elif (progress - center) > 0:
            # right side laps
            positions[0] = center + np.sin((progress - center - 2 * lap_length) / r)
            positions[1] = -r - np.cos((progress - center - 2 * lap_length) / r)
        elif progress < center:
            # initial straight
            positions[0] = progress
            positions[1] = 0
        return positions

    @staticmethod
    # @numba.njit
    def pos2progress(x, y, lap):
        """turns a position and lap count to path progress"""
        progress = 0
        if lap == 0:
            progress = x
        elif (0 < lap) and (lap < 3):
            angle = -np.arctan2(x - center, -r - y) + np.pi
            progress = angle * r + (lap - 1) * lap_length + center
        elif (2 < lap) and (lap < 5):
            angle = np.arctan2(center - x, y - r) + np.pi
            progress = angle * r + (lap - 1) * lap_length + center
        else:
            progress = x + 4 * lap_length
        return progress

    def request_progress(self, x, y, lap):
        current_progress = self.pos2progress(x, y, lap)
        lap_correct = True
        if self.prev_progress > (current_progress + 2 * r):  # the pi is not missing
            lap += 1
            current_progress = self.pos2progress(x, y, lap)
            lap_correct = False
        return current_progress, lap_correct

    def request_waypoints(self, x, y, heading, lap, Nt=-1, dt=-1):
        if Nt == -1:
            Nt = self.Nt
        if dt == -1:
            dt = self.dt
            prev_progress = self.prev_progress

        current_progress = self.pos2progress(x, y, lap)
        if prev_progress > (current_progress + 2 * r):  # the pi is not missing
            lap += 1
            current_progress = self.pos2progress(x, y, lap)
        progresses = np.zeros((Nt + 1,))
        waypoints = np.zeros((Nt + 1, 2))
        speeds = np.zeros((Nt + 1,))

        progresses[0] = current_progress
        waypoints[0, :] = [x, y]

        if lap in range(0, 8):
            speeds = self.target_vel * np.ones((Nt + 1))
            progresses[1:] = progresses[0] + np.cumsum(speeds[:-1]) * dt
        else:
            for i in range(Nt):
                speeds[i] = min(self.progress2speed(progresses[i]), self.target_vel)
                speeds[i] = min(
                    self.progress2speed(progresses[i] + speeds[i] * dt / 2),
                    self.target_vel,
                )  # overwrites prvious line?
                progresses[i + 1] = progresses[i] + speeds[i] * dt

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

    def request_desired_speed(self, progress) -> float:
        if progress < self.braking_zone_1_start:
            return self.target_vel
        elif progress < self.braking_zone_2_start:
            return self.target_vel * self.slowdown_speed_factor
        else:
            return 0


def test_planning():
    Nt = 500
    tf = 5

    dt = tf / Nt

    planner = SkidpadPlanner(target_vel=9.0, Nt=Nt, dt=dt)

    x1 = 18
    y1 = 0
    lap1 = 5
    [target_positions, _, _, _] = planner.request_waypoints(x1, y1, 0, lap1)

    waypoints = target_positions
    print(f"Waypoints: {waypoints}")
    circle1 = plt.Circle(
        (center, -r),
        radius=r,  # - 1.5,
        color="blue",
        fill=False,
    )
    circle2 = plt.Circle(
        (center, r),
        radius=r,
        color="blue",
        fill=False,
    )

    fig, ax = plt.subplots()

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    colors = np.arange(waypoints.shape[0]) + 3
    plt.scatter((waypoints[:, 0]), (waypoints[:, 1]), c=colors, cmap="Reds")
    plt.xlim([0, 50])
    plt.ylim([-25, 25])
    plt.scatter(x1, y1)
    np.save("waypoints", waypoints)
    print(waypoints[:, 2:])
    plt.figure()
    plt.scatter(waypoints[:, 0], waypoints[:, 1])
    plt.plot([0], [0], "or")
    plt.show()


if __name__ == "__main__":
    test_planning()
    planner = SkidpadPlanner(2)
    print(planner.progress2speed(31.85))
    print(SkidpadPlanner.progress2position(31.25))
