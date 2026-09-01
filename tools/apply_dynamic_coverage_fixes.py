from __future__ import annotations

from pathlib import Path

PATH = Path("src/eyeprocesspy/dynamic_irt.py")
text = PATH.read_text(encoding="utf-8")

replacements = {
    'if d[[person]].isna().any() or d[[item]].isna().any():':
        'if d[person].isna().any() or d[item].isna().any():',
    '    for prefix in variables: keep|=vals.str.startswith(prefix).to_numpy(); return s.loc[keep].copy()':
        '    for prefix in variables:\n        keep |= vals.str.startswith(prefix).to_numpy()\n    return s.loc[keep].copy()',
}

for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one occurrence of {old!r}; found {count}.")
    text = text.replace(old, new, 1)

PATH.write_text(text, encoding="utf-8")
print("DYNAMIC_IRT_EXACT_FIXES_APPLIED=2")
