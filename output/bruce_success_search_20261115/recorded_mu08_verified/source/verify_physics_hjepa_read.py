#!/usr/bin/env python3
"""Read actual RGB physics records through H-JEPA's loader; does not train a model."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stable_worldmodel.data.dataset import HDF5Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--names',nargs='+',default=['stand','box08'])
    args = parser.parse_args()
    reports = []
    for name in args.names:
        dataset = HDF5Dataset(name=name, cache_dir=args.directory,
            keys_to_load=['pixels', 'proprio', 'action'], keys_to_cache=['action'],
            level1=dict(frameskip=2, num_steps=4))
        sample = dataset[0]
        assert sample['pixels_level1'].shape == (4, 3, 64, 64)
        assert sample['proprio_level1'].shape == (4, 36)
        assert sample['action_level1'].shape == (4, 128)
        ep, start = dataset.clip_indices[0]
        offset = int(dataset.offsets[ep])+start
        np.testing.assert_array_equal(np.asarray(sample['action_level1']).reshape(8, 64),
                                      dataset._cache['action'][offset:offset+8])
        reports.append(dict(dataset=name, frames=int(dataset.lengths.sum()),
            clips=len(dataset), sample_shapes={k:list(v.shape) for k,v in sample.items()},
            action_alignment_passed=True, model_training_performed=False))
        dataset.h5_file.close()
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
