"""Check diagnostic read compatibility/failure fallback; no serial access."""
from thermal_capture_research import make_wrapper, BASE_FIELDS, EXTENDED_FIELDS

events=[];calls=[]
def normal(_self, ids, fields):
    calls.append((ids,tuple(fields)))
    if tuple(fields)==EXTENDED_FIELDS:
        return [([16.,69.,65.,62.],128),([16.1,71.,68.,65.],128)]
    return [([16.,69.],128),([16.1,71.],128)]

wrapped=make_wrapper(normal,events.append,now=lambda:42.)
actual=wrapped(object(),[4,9],list(BASE_FIELDS))
assert actual==[([16.,69.],128),([16.1,71.],128)]
assert events[0]['attention_70c'] and events[0]['monotonic']==42.
assert events[0]['maximum_component_temperature_c']==71.
assert calls==[([4,9],EXTENDED_FIELDS)]
wrapped(object(),[1,2],list(BASE_FIELDS))
assert calls[-1]==([1,2],BASE_FIELDS) and len(events)==1

for bad in [None,[None,([16.,71.,68.,65.],128)],
            [([16.,float('nan'),65.,62.],128),([16.1,71.,68.,65.],128)]]:
    calls=[];events=[]
    def failure(_self,ids,fields):
        calls.append(tuple(fields))
        if tuple(fields)==EXTENDED_FIELDS:return bad
        return [([16.,69.],128),([16.1,71.],128)]
    assert make_wrapper(failure,events.append)(object(),[4,9],list(BASE_FIELDS))==[([16.,69.],128),([16.1,71.],128)]
    assert calls==[EXTENDED_FIELDS,BASE_FIELDS] and events[0]['status']=='extended_read_failed'

def bad_sink(_):raise OSError('disk error')
assert make_wrapper(normal,bad_sink)(object(),[4,9],list(BASE_FIELDS))==[([16.,69.],128),([16.1,71.],128)]
print('PASS field shape/status preserved, 70C notification, malformed/failed reads and log failures; no motor IO')
