from pathlib import Path
import inspect
import eyeprocesspy as ep


def test_frozen_0_9_irt_exports_are_public_real_callables():
    names=[x.strip() for x in (Path(__file__).parent/'fixtures'/'irt_0_9_exports.txt').read_text().splitlines() if x.strip()]
    assert len(names)==115
    missing=[n for n in names if not hasattr(ep,n)]
    assert not missing, missing
    bad=[]
    for name in names:
        obj=getattr(ep,name)
        if not callable(obj): bad.append((name,'not callable'))
        elif inspect.isfunction(obj):
            src=inspect.getsource(obj)
            if 'NotImplementedError' in src or src.strip().endswith('pass'): bad.append((name,'placeholder'))
    assert not bad, bad
