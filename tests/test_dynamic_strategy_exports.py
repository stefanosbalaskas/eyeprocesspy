from pathlib import Path
import inspect
import eyeprocesspy as ep


def test_dynamic_strategy_frozen_exports_are_public_and_real():
    names=[x.strip() for x in (Path(__file__).parent/'fixtures'/'dynamic_strategy_diffusion_exports.txt').read_text().splitlines() if x.strip()]
    assert len(names)==35
    missing=[n for n in names if not hasattr(ep,n)]
    assert not missing
    for name in names:
        obj=getattr(ep,name)
        assert callable(obj), name
        if inspect.isfunction(obj):
            src=inspect.getsource(obj)
            assert 'NotImplementedError' not in src
            assert src.strip() not in {f'def {name}():\n    pass'}
