"""Regress the false summary timeout while retaining real feedback timeouts."""
from preparation_feedback_guard import FeedbackGuard

# 2-second summaries may be slightly older than 2 s. They are not a sensor
# stream. Continuous live signals must pass independently of summary cadence.
g=FeedbackGuard()
for i in range(250):
    now=100+i*.04
    summary_time=now-(i%53)*.04  # includes summary ages > 2 seconds
    assert now-summary_time < 2.2
    g.check(now,1000+now,1000+now-.01,1000+now-.005,now-.04,i*.04)

for key in ['sense','leg','motor','estimator']:
    g=FeedbackGuard()
    g.check(100.,1100.,1099.99,1099.995,99.96,5.)
    kwargs=dict(now_mono=100.16,now_wall=1100.16,sense_wall=1100.15,
                leg_wall=1100.15,motor_mono=100.12,estimator_progress=5.16)
    if key=='sense':kwargs['sense_wall']=1099.99
    if key=='leg':kwargs['leg_wall']=1099.995
    if key=='motor':kwargs['motor_mono']=99.95
    if key=='estimator':kwargs['estimator_progress']=5.
    try:g.check(**kwargs)
    except RuntimeError as error:assert 'stale' in str(error)
    else:raise AssertionError('Real stale feedback accepted: '+key)
print('PASS: slow summaries do not stop control; actual IMU, motor, leg and estimator timeouts remain enforced')
