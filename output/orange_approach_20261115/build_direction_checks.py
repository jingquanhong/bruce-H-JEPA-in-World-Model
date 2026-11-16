from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'test_imu_heading_high_private.py').read_text()
s=s.replace('import Play.Walking.high_level as baseline','from Util import imu_heading_high_level as baseline')
s=s.replace('imu_heading_high_level as candidate','imu_direction_high_level as candidate')
s=s.replace("for sign in (0.,1.):","for vx,vy in ((0.,0.),(.08,0.),(-.08,0.),(0.,.03),(0.,-.03)):")
s=s.replace("np.array([sign*.05,0.])","np.array([vx,vy])")
s=s.replace("profile=label,requested_vx=sign*.05,","profile=label,requested_vx=vx,requested_vy=vy,")
s=s.replace('(label,sign,actual_phase,expected_phase)','(label,vx,vy,actual_phase,expected_phase)')
a=s.index('# With zero forward command')
s=s[:a]+'''# Stationary, forward and backward behavior must match proven planner exactly.
for vx in (0.,.08,-.08):
 a=next(r['first_command'] for r in reports if r['profile']=='baseline' and r['requested_vx']==vx and r['requested_vy']==0.)
 b=next(r['first_command'] for r in reports if r['profile']=='heading' and r['requested_vx']==vx and r['requested_vy']==0.)
 for key in a:assert np.allclose(a[key],b[key],atol=1e-12), (vx,key)
(root/'imu-direction-private-results.json').write_text(json.dumps(dict(result='PASS_PRIVATE_SOFTWARE_ONLY',cases=reports,note='No actuation or physical dynamics; lateral physical response unvalidated.'),indent=2))
print('PASS: unchanged stationary/forward/back initial output; lateral planner finite in both directions. Private software only.')
'''
s=s.replace('imu-heading-private-planner.log','imu-direction-private-planner.log')
(p/'test_imu_direction_high_private.py').write_text(s)
