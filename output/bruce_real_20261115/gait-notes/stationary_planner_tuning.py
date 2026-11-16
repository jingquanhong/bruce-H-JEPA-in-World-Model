"""Bounded, per-step stationary planner tuning; contains no actuator access."""
import json
import math
from pathlib import Path


def checked_tuning(forward_bias_m, minimum_width_m, lateral_bias_m=0.):
    bias, width, lateral = float(forward_bias_m), float(minimum_width_m), float(lateral_bias_m)
    if not math.isfinite(bias) or not -.02 <= bias <= .02:
        raise ValueError('Forward DCM bias must be -0.02..0.02 meters')
    if not math.isfinite(width) or not .04 <= width <= .12:
        raise ValueError('Minimum lateral step width must be 0.04..0.12 meters')
    if not math.isfinite(lateral) or not -.02 <= lateral <= .02:
        raise ValueError('Lateral DCM bias must be -0.02..0.02 meters')
    return bias, width, lateral


def load_tuning(path, default_bias, default_width):
    if path is None:
        return default_bias, default_width, 0.
    data = json.loads(Path(path).read_text())
    return checked_tuning(data['forward_bias_m'], data['minimum_width_m'], data.get('lateral_bias_m',0.))
