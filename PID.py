import time
import numpy as np

class PID:
    def __init__(self, kp: float = 1.0, ki: float = 1.0, kd: float = 1.0,
                      u_min: float = None, u_max: float = None, t_aw: float = None):
        # regulator variables
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_value = 0.0
        self._e_old = 0.0

        # anti-windup variables
        self._u_min = u_min
        self._u_max = u_max
        self._t_aw = 1/ki if(t_aw is None) else t_aw
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
        d = (self._kd * error_diff) if(error_diff is not None) else 0.0

        u_temp = p + i + d          # unsaturaded control output

        # output clipping (saturated control output)
        u = u_temp
        if(self._u_max is not None):
            u = min(u, self._u_max)
        if(self._u_min is not None):
            u = max(u, self._u_min)

        # back-calculation anti-windup
        e_track = u - u_temp # saturation error
        self._integral_value += (error + (1.0 / self._t_aw) * e_track) * dt    # subtracting the saturation error

        self._e_old = error
        self._tp = curr_t
        self._u_old = u

        return u
