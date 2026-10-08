import time
import numpy as np

class PID:
    def __init__(self, kp: float = 1.0, ki: float = 1.0, kd: float = 1.0,
                      u_min: float = 0.0, u_max: float = 0.0):
        # regulator variables
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_value = 0.0
        self._e_old = 0.0

        # anti-windup variables
        self._u_min = u_min
        self._u_max = u_max
        self._u_old = 0.0

        # additional variables
        self._tp = time.monotonic()

    def solve(self, error: float, error_diff: float = None) -> float:
        curr_t = time.monotonic()
        dt = curr_t - self._tp
        if(dt <= 0.0):
            return self._u_old

        p = self._kp * error
        i = self._ki * self._integral_value
        d = (self._kd * error_diff) if(error_diff is not None) else 0
        u = p + i + d

        if not ((u >= self._u_max and error > 0.0) or (u <= self._u_min and error < 0.0)):
            self._integral_value += error * dt

        i = self._ki * self._integral_value
        u = p + i + d
        u = np.clip(u, self._u_min, self._u_max)

        self._e_old = error
        self._tp = curr_t
        self._u_old = u

        return u
