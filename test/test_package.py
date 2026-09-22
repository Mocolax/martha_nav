from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_package_imports():
    import martha_nav.learning  # noqa: F401
    import martha_nav.sim2d  # noqa: F401


def test_worlds_copied():
    names = sorted(p.stem for p in (REPO / 'worlds').glob('*.world'))
    assert names == ['four_rooms', 'hall', 'lab', 'multi', 'roblab', 'room', 'tube']
