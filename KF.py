import numpy as np
import time

class KalmanFilter():
    def __init__(self, dt: float = 0.05, B: np.ndarray = None, Q: np.ndarray= None, R: np.ndarray= None,
                  x0: np.ndarray = np.zeros((6, 1), dtype = np.float32), P0: np.ndarray = np.eye(6, dtype=np.float32), sigma: float = 1.0):
        self._F = np.array([[1, 0, 0, dt, 0, 0],
                             [0, 1, 0, 0, dt, 0],
                             [0, 0, 1, 0, 0, dt],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]], dtype= np.float32)
        self._B = B if(B is not None) else None                     # no known input
        self._H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0]], dtype= np.float32)
        if(Q is not None):
            self._Q = Q
        else:
            self._get_Q_cv(dt=dt, sigma_a=sigma)
        self._R = R if(R is not None) else np.eye(3, dtype=np.float32)
        self._x = x0
        self._P = P0
        self._sigma_a = sigma
        self._tp = time.monotonic()

    def predict(self, u: np.ndarray = None) -> np.ndarray:
        t_n = time.monotonic()
        dt = t_n - self._tp
        self._tp = t_n

        self._F[0, 3] = dt
        self._F[1, 4] = dt
        self._F[2, 5] = dt
        self._get_Q_cv(dt, self._sigma_a)

        self._x = self._F @ self._x
        if self._B is not None:
            self._x += self._B @ u
        self._P = self._F @ (self._P @ self._F.T) + self._Q

        return self._x

    def update(self, z: np.ndarray) -> np.ndarray:
        S = self._H @ (self._P @ self._H.T) + self._R
        K = (self._P @ self._H.T) @ np.linalg.inv(S)
        y = z - self._H @ self._x

        self._x = self._x + K @ y
        I = np.eye(self._P.shape[0], dtype=np.float32)
        self._P = (I - K @ self._H) @ self._P 

        return self._x

    def _get_Q_cv(self, dt: float, sigma_a: float):
        self._Q = np.array([[(dt**3) / 3.0, 0, 0, (dt**2) / 2.0, 0, 0],
                            [0, (dt**3) / 3.0, 0, 0, (dt**2) / 2.0, 0],
                            [0, 0, (dt**3) / 3.0, 0, 0, (dt**2) / 2.0],
                            [(dt**2) / 2.0, 0, 0, dt, 0, 0],
                            [0, (dt**2) / 2.0, 0, 0, dt, 0],
                            [0, 0, (dt**2) / 2.0, 0, 0, dt],], dtype=np.float32) * (sigma_a ** 2)
