"""Slow, double-support preparation only. This code cannot request a jump."""
import math


def checked_depth(value):
    value = float(value)
    if not math.isfinite(value) or not .005 <= value <= .015:
        raise ValueError("Preparation depth must be 0.005..0.015 m")
    return value


def smooth(value):
    u = min(1., max(0., value))
    return u**3 * (10. - 15.*u + 6.*u*u)


def offset_at(elapsed, depth):
    depth = checked_depth(depth)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid elapsed time")
    if elapsed < 2.:return -depth * smooth(elapsed / 2.)
    if elapsed < 2.5:return -depth
    if elapsed < 4.5:return -depth * (1. - smooth((elapsed - 2.5) / 2.))
    return 0.


class PreparationCycle:
    def __init__(self, now, depth, initial_com_z):
        if not math.isfinite(now) or not math.isfinite(initial_com_z):
            raise ValueError("Invalid start state")
        self.started = now
        self.depth = checked_depth(depth)
        self.initial_z = initial_com_z
        self.minimum_z = initial_com_z
        self.maximum_tilt = 0.
        self.last_time = now
        self.contact_lost = None

    def update(self, now, com_z, tilt, contacts):
        if not all(math.isfinite(v) for v in [now, com_z, tilt]) or now < self.last_time:
            raise RuntimeError("Invalid preparation feedback/time")
        if len(contacts) != 4 or not all(v in (0, 1) for v in contacts):
            raise RuntimeError("Invalid contact feedback")
        self.last_time = now
        self.minimum_z = min(self.minimum_z, com_z)
        self.maximum_tilt = max(self.maximum_tilt, tilt)
        if tilt > 8.:raise RuntimeError("Preparation tilt exceeded 8 degrees")
        grounded = any(contacts[:2]) and any(contacts[2:])
        if grounded:self.contact_lost = None
        elif self.contact_lost is None:self.contact_lost = now
        elif now - self.contact_lost > .1:
            raise RuntimeError("Unexpected loss of double support during slow preparation")
        elapsed = now - self.started
        return offset_at(elapsed, self.depth), elapsed >= 6.5

    def result(self, final_z):
        excursion = self.initial_z - self.minimum_z
        recovered = abs(final_z - self.initial_z) < max(.002, self.depth * .3)
        response = excursion >= self.depth * .4
        return dict(result="PREPARATION_RESPONSE_SEEN" if response and recovered else "PREPARATION_RESPONSE_INCOMPLETE",
                    commanded_crouch_depth_m=self.depth, estimated_com_excursion_m=excursion,
                    estimated_final_z_error_m=final_z-self.initial_z,
                    maximum_tilt_deg=self.maximum_tilt, real_jump_attempted=False,
                    ground_truth=False, feet_expected_in_contact=True)
