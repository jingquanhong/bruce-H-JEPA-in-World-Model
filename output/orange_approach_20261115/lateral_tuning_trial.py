"""Temporary, bounded planner tuning for a supported lateral trial.

No actuator or shared-memory access. Restore only the file this trial changed.
"""
import json
import math
import os
from pathlib import Path


def atomic(path, data):
    temp = path.with_suffix('.new')
    temp.write_text(json.dumps(data))
    os.replace(temp, path)


class LateralTuningTrial:
    def __init__(self, path, offset):
        self.path = Path(path)
        self.offset = float(offset)
        if not math.isfinite(self.offset) or abs(self.offset) > .006:
            raise ValueError('Trial lateral bias offset must be within +/-0.006 m')
        self.before = self.during = None
        self.status = 'not_applied'

    def apply(self):
        self.before = json.loads(self.path.read_text())
        self.during = dict(self.before)
        value = float(self.before['lateral_bias_m']) + self.offset
        if not math.isfinite(value) or not -.02 <= value <= .02:
            raise ValueError('Resulting lateral bias exceeds original planner bounds')
        self.during['lateral_bias_m'] = value
        if self.offset:
            atomic(self.path, self.during)
            self.status = 'applied'
        else:
            self.status = 'unchanged'

    def restore(self):
        if self.status != 'applied':
            return
        if not self.path.exists():
            self.status = 'runtime_removed_after_stop'
            return
        if json.loads(self.path.read_text()) != self.during:
            self.status = 'external_change_restore_refused'
            raise RuntimeError('Planner tuning changed externally; refusing to overwrite it')
        atomic(self.path, self.before)
        self.status = 'restored'

    def record(self):
        return dict(offset_m=self.offset, before=self.before, during=self.during,
                    status=self.status)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'planner-tuning.json'
        baseline = dict(forward_bias_m=-.002, minimum_width_m=.10, lateral_bias_m=-.006)
        atomic(path, baseline)
        trial = LateralTuningTrial(path, .002)
        try:
            trial.apply()
            assert json.loads(path.read_text()) == dict(baseline, lateral_bias_m=-.004)
            raise InterruptedError('simulate interrupted observation')
        except InterruptedError:
            pass
        finally:
            trial.restore()
        assert json.loads(path.read_text()) == baseline
        assert trial.status == 'restored'
        for bad in (float('nan'), float('inf'), .00601, -.00601):
            try:
                LateralTuningTrial(path, bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid offset accepted')
        trial = LateralTuningTrial(path, .002)
        trial.apply()
        atomic(path, dict(baseline, lateral_bias_m=0.))
        try:
            trial.restore()
        except RuntimeError:
            pass
        else:
            raise AssertionError('External edit overwritten')
        assert json.loads(path.read_text())['lateral_bias_m'] == 0.
    print('PASS lateral tuning bounds, interruption restoration, external edit protection')


if __name__ == '__main__':
    self_test()
