from pathlib import Path
import inspect
import eyeprocesspy as ep

def test_frozen_measurement_intelligence_exports_are_public_real_callables():
    names=[x.strip() for x in (Path(__file__).parent/'fixtures'/'measurement_intelligence_exports.txt').read_text().splitlines() if x.strip()]
    assert len(names)==35
    assert not [n for n in names if not hasattr(ep,n)]
    assert not [n for n in names if not callable(getattr(ep,n))]
    for n in names:
        obj=getattr(ep,n)
        if inspect.isfunction(obj):
            src=inspect.getsource(obj)
            assert 'NotImplementedError' not in src
