import numpy as np
from src.region import Region, create_root_region, split_regions_once


def test_region_center_and_volume():
    r = create_root_region()
    c = r.center()
    assert np.allclose(c, (0.5, 0.5))
    assert abs(r.volume() - 1.0) < 1e-12


def test_contains_and_sample():
    r = create_root_region()
    rng = np.random.default_rng(0)
    pts = r.sample_uniform(10, rng)
    mask = r.contains(pts)
    assert mask.all()


def test_split_returns_four_children():
    r = create_root_region()
    children = r.split()
    assert len(children) == 4
    for i, c in enumerate(children):
        assert c.parent_id == r.region_id
        assert c.depth == r.depth + 1
