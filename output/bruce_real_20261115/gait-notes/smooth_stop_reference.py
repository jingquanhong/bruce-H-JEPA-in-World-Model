"""Continuous CoM and body-orientation references for a supported stop."""
import numpy as np
from Util import math_function as mf


class SmoothStopReference:
    def __init__(self, p0, v0, p1, R0, R1, duration=.8):
        self.p0, self.v0, self.p1 = [np.asarray(x, dtype=float).copy() for x in (p0, v0, p1)]
        self.R0, self.R1 = [np.asarray(x, dtype=float).copy() for x in (R0, R1)]
        self.duration = float(duration)
        if not .2 <= self.duration <= 2.:
            raise ValueError('Stop transition duration must be 0.2..2 seconds')
        if any(x.shape != (3,) for x in (self.p0, self.v0, self.p1)) or any(x.shape != (3, 3) for x in (self.R0, self.R1)):
            raise ValueError('Invalid stop reference dimensions')
        if not all(np.isfinite(x).all() for x in (self.p0, self.v0, self.p1, self.R0, self.R1)):
            raise ValueError('Nonfinite stop reference')
        delta = self.p1 - self.p0
        velocity = self.v0 * self.duration
        self.c3 = 10 * delta - 6 * velocity
        self.c4 = -15 * delta + 8 * velocity
        self.c5 = 6 * delta - 3 * velocity
        self.rotation = mf.logvee(self.R0.T @ self.R1)

    def sample(self, elapsed):
        if not np.isfinite(elapsed):
            raise ValueError('Nonfinite transition time')
        s = float(np.clip(elapsed / self.duration, 0., 1.))
        p = self.p0 + self.v0 * self.duration * s + self.c3*s**3 + self.c4*s**4 + self.c5*s**5
        v = self.v0 + (3*self.c3*s**2 + 4*self.c4*s**3 + 5*self.c5*s**4) / self.duration
        blend = 10*s**3 - 15*s**4 + 6*s**5
        rate = (30*s**2 - 60*s**3 + 30*s**4) / self.duration
        R = self.R0 @ mf.hatexp(self.rotation * blend)
        omega = self.rotation * rate
        return p, v, R, omega
