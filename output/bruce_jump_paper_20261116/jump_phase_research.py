"""Experimental, offline-only contact phase classifier. No robot/motor IO.

Thresholds are research assumptions, not calibrated timing or authorization to
jump. Outputs are labels for replay and must never be used to suppress safety
checks. In particular, the current stance controller must not consume them.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Sample:
    time_s: float
    contacts: tuple
    com_vz_m_s: float
    tilt_deg: float
    sense_age_s: float
    leg_age_s: float
    motor_age_s: float
    motor_fault: bool = False
    emergency_stop: bool = False


class JumpPhaseReplay:
    """Distinguish candidate push, flight, landing and failure in log replay."""

    def __init__(self):
        self.phase = 'ready'
        self.started = self.last_time = self.candidate_since = None
        self.flight_started = self.landing_started = None
        self.abort_reason = None

    def abort(self, reason):
        self.phase = 'abort'
        self.abort_reason = reason
        self.candidate_since = None

    def start_push(self, now):
        if self.phase != 'ready' or not math.isfinite(now):
            raise ValueError('Can start only from ready with a finite time')
        self.phase = 'push'
        self.started = self.last_time = now

    def update(self, sample):
        if self.phase in ('abort', 'landed'):
            return self.phase
        if self.phase == 'ready':
            raise ValueError('Push has not been started')
        fields = (sample.time_s, sample.com_vz_m_s, sample.tilt_deg,
                  sample.sense_age_s, sample.leg_age_s, sample.motor_age_s)
        if not all(math.isfinite(v) for v in fields):
            self.abort('nonfinite feedback')
            return self.phase
        if sample.time_s < self.last_time or sample.time_s - self.last_time > .15:
            self.abort('feedback time discontinuity')
        elif (not 0 <= sample.sense_age_s <= .15 or
              not 0 <= sample.leg_age_s <= .15 or
              not 0 <= sample.motor_age_s <= .2):
            self.abort('stale feedback')
        elif sample.emergency_stop or sample.motor_fault:
            self.abort('emergency stop or motor fault')
        elif not 0 <= sample.tilt_deg <= 8:
            self.abort('tilt bound exceeded')
        elif len(sample.contacts) != 4 or any(x not in (0, 1) for x in sample.contacts):
            self.abort('invalid contacts')
        if self.phase == 'abort':
            return self.phase
        self.last_time = sample.time_s
        right = bool(sample.contacts[0] or sample.contacts[1])
        left = bool(sample.contacts[2] or sample.contacts[3])
        if self.phase == 'push':
            if sample.time_s - self.started > .5:
                self.abort('no confirmed liftoff')
                return self.phase
            candidate = not right and not left and sample.com_vz_m_s > .05
            self.candidate_since = (self.candidate_since if self.candidate_since is not None else sample.time_s) if candidate else None
            if self.candidate_since is not None and sample.time_s - self.candidate_since >= .015:
                self.phase = 'flight'
                self.flight_started = sample.time_s
                self.candidate_since = None
        elif self.phase == 'flight':
            if sample.time_s - self.flight_started > .5:
                self.abort('flight timeout')
            elif right or left:
                self.phase = 'landing'
                self.landing_started = sample.time_s
        elif self.phase == 'landing':
            if sample.time_s - self.landing_started > .4:
                self.abort('landing timeout')
            else:
                self.candidate_since = (self.candidate_since if self.candidate_since is not None else sample.time_s) if right and left else None
                if self.candidate_since is not None and sample.time_s - self.candidate_since >= .05:
                    self.phase = 'landed'
        return self.phase
