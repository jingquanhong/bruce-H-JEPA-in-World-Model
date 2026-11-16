"""Opt-in thermal capture decorator for existing BEAR diagnostic reads.

No serial interfaces or motor control are opened here. The wrapper only extends
the driver's existing two-knee diagnostic read; a failed extended read falls
back to the exact original read. Never used by the validated walking version.
This is instrumentation, not an autonomous stop or complete thermal safety.
"""
import json
import math
import os
from pathlib import Path
import time


BASE_FIELDS = ('input_voltage', 'winding_temperature')
EXTENDED_FIELDS = BASE_FIELDS + ('powerstage_temperature', 'ic_temperature')
KNEE_IDS = (4, 9)


def make_wrapper(original, sink, now=time.monotonic):
    """Return wrapped bulk_read while preserving original return data shape."""
    def read(self, *args, **kwargs):
        if len(args) != 2 or kwargs or tuple(args[0]) != KNEE_IDS or tuple(args[1]) != BASE_FIELDS:
            return original(self, *args, **kwargs)
        try:
            response = original(self, args[0], list(EXTENDED_FIELDS))
            if response is None or len(response) != 2 or any(x is None or len(x) != 2 for x in response):
                raise ValueError('incomplete extended diagnostic read')
            if any(len(x[0]) != len(EXTENDED_FIELDS) for x in response):
                raise ValueError('unexpected register count')
            if not all(math.isfinite(float(v)) for item in response for v in item[0]):
                raise ValueError('nonfinite temperature or voltage')
        except Exception as exc:
            try:sink({'monotonic':now(), 'status':'extended_read_failed', 'error':repr(exc)})
            except Exception:pass
            return original(self, *args, **kwargs)
        record={'monotonic':now(), 'status':'ok', 'ids':list(KNEE_IDS),
                'registers':list(EXTENDED_FIELDS),
                'values':[[list(x[0]),x[1]] for x in response]}
        temperatures=[float(v) for row in response for v in row[0][1:]]
        record['maximum_component_temperature_c']=max(temperatures)
        record['attention_70c']=record['maximum_component_temperature_c'] >= 70.
        try:sink(record)
        except Exception:pass  # logging must not change the existing driver response
        short=[(list(item[0][:len(BASE_FIELDS)]), item[1]) for item in response]
        return type(response)(short) if isinstance(response, tuple) else short
    return read


def file_sink(folder):
    folder=Path(folder)
    def publish(record):
        temp=folder/'knee-temperatures.new'
        temp.write_text(json.dumps(record))
        os.replace(temp,folder/'knee-temperatures.json')
        with (folder/'knee-temperatures.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        if record.get('attention_70c'):
            print('KNEE_TEMPERATURE_ATTENTION_70C', record['values'],flush=True)
    return publish
