"""Pure timestamp checks for a read-only observer. No hardware imports."""
import math


def feedback_checks(now,wall_now,motor_stamp,sensor_stamp,estimator_age,
                    estimator_stamp,codes,rotation_valid,voltage,values_valid=True):
    def fresh(age,limit):return math.isfinite(age) and 0<=age<limit
    return dict(motor_codes=codes==[128]*10,
        motor_fresh=fresh(now-motor_stamp,.2),
        sensor_fresh=fresh(wall_now-sensor_stamp,.15),
        estimator_progress=math.isfinite(estimator_stamp) and fresh(estimator_age,.15),
        orientation_valid=bool(rotation_valid),motor_values_valid=bool(values_valid),
        voltage_ready=voltage is not None and math.isfinite(voltage) and voltage>15.)


def self_test():
    # Actual race from the previous trial: the read returned a snapshot newer
    # than the loop's starting timestamp. Sampling completion is the right time.
    start=756.166696462;motor=756.167091955;end=756.168;wall=1800000000.
    def check(now=end,stamp=motor,sensor=wall-.01,est_age=.01,est_stamp=179.,
              codes=None,rotation=True,voltage=15.57):
        return feedback_checks(now,wall,stamp,sensor,est_age,est_stamp,
                               [128]*10 if codes is None else codes,rotation,voltage)
    assert not all(check(now=start).values())
    assert all(check().values())
    assert not check(stamp=end-.201)['motor_fresh']
    assert not check(stamp=end+.001)['motor_fresh']
    assert not check(sensor=wall-.151)['sensor_fresh']
    assert not check(est_age=.151)['estimator_progress']
    assert not check(est_stamp=float('nan'))['estimator_progress']
    for relative_stamp in (0.,.1,179.,1e6):assert all(check(est_stamp=relative_stamp).values())
    assert not check(codes=[136]*10)['motor_codes']
    assert not check(rotation=False)['orientation_valid']
    assert not check(voltage=14.)['voltage_ready']
    assert not check(voltage=None)['voltage_ready']
    print('Feedback clock regression passed: real snapshot race, relative estimator time, stale/future data, invalid feedback.')


if __name__=='__main__':self_test()
