import numpy as np

from martha_nav.ros.world_map_publisher import load_map_yaml

# 3 wide, 2 tall, as map_saver writes it: top row first; 254 free, 0 occupied, 205 unknown.
PGM = b'P5\n# CREATOR: map_saver.cpp 0.050 m/pix\n3 2\n255\n' + bytes([0, 254, 205, 254, 254, 254])
YAML = 'image: lab_real.pgm\nmode: trinary\nresolution: 0.05\norigin: [-1.5, -0.5, 0]\n' \
       'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n'


def test_saved_map_becomes_a_grid(tmp_path):
    (tmp_path / 'lab_real.pgm').write_bytes(PGM)
    (tmp_path / 'lab_real.yaml').write_text(YAML)
    grid = load_map_yaml(tmp_path / 'lab_real.yaml')
    assert grid.origin == (-1.5, -0.5) and grid.resolution == 0.05
    # Row 0 is the bottom of the image; the occupied and the unknown pixel block.
    assert np.array_equal(grid.occ, [[False, False, False], [True, False, True]])
