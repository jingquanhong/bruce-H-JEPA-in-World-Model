"""Exercise WBC startup using private memory; never open an actuator port."""
import types
import numpy as np


def warm_low_level(module, fixture, capture=None):
    import Settings.BRUCE_data as rds

    class Completed(Exception):
        pass

    class PrivateSegment:
        def __init__(self, name, data):
            self.name = name
            self.data = {k: np.asarray(v, dtype=float).copy() for k, v in data.items()}
            self.command_count = 0

        def get(self):
            return {k: v.copy() for k, v in self.data.items()}

        def set(self, data, opt=None):
            for k, v in data.items():
                if k in self.data:
                    self.data[k] = np.asarray(v).copy()
            if self.name == 'LEG_COMMAND' and 'goal_torques' in data and data.get('BEAR_enable', [0])[0] == 1:
                if not all(np.isfinite(np.asarray(v)).all() for v in data.values()):
                    raise RuntimeError('Nonfinite WBC warmup command')
                if capture is not None:
                    capture.append({k: np.asarray(v).copy() for k,v in data.items()})
                self.command_count += 1
                if self.command_count >= 3:
                    raise Completed()

    private = types.SimpleNamespace(**{name: PrivateSegment(name, values) for name, values in fixture.items()})
    private.THREAD_STATE.set({k: np.zeros(1) for k in private.THREAD_STATE.data})
    private.THREAD_STATE.set({'estimation': np.ones(1)})
    previous = module.MM, rds.MM, rds.safe_input
    try:
        module.MM = private
        rds.MM = private
        rds.safe_input = lambda prompt: 'y'
        try:
            module.main_loop()
        except Completed:
            pass
        else:
            raise RuntimeError('WBC warmup exited before private command validation')
        return private.LEG_COMMAND.command_count
    finally:
        module.MM, rds.MM, rds.safe_input = previous
