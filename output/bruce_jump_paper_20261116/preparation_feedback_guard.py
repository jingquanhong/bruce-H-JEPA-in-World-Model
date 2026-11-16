"""Check critical feedback freshness independently of slow status summaries."""
import math


class FeedbackGuard:
    def __init__(self):
        self.estimator_progress = None
        self.estimator_changed = None

    def check(self, now_mono, now_wall, sense_wall, leg_wall, motor_mono,
              estimator_progress):
        values = [now_mono, now_wall, sense_wall, leg_wall, motor_mono, estimator_progress]
        if not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError('Nonfinite feedback timestamp')
        if self.estimator_progress is None or estimator_progress != self.estimator_progress:
            if self.estimator_progress is not None and estimator_progress < self.estimator_progress:
                raise RuntimeError('Estimator clock restarted during observation')
            self.estimator_progress = estimator_progress
            self.estimator_changed = now_mono
        ages = {'sense_s': now_wall-sense_wall, 'leg_s': now_wall-leg_wall,
                'motor_s': now_mono-motor_mono,
                'estimator_stalled_s': now_mono-self.estimator_changed}
        limits = {'sense_s': .15, 'leg_s': .15, 'motor_s': .2,
                  'estimator_stalled_s': .15}
        for key, limit in limits.items():
            if ages[key] < -.02:
                raise RuntimeError('Feedback timestamp is in the future: '+key)
            if ages[key] > limit:
                raise RuntimeError('Critical feedback stale: '+key+'='+str(ages[key]))
        return ages
