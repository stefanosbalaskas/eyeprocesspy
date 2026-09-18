# Pixels versus Degrees of Visual Angle

Pixels describe display coordinates. Degrees of visual angle describe angular size at the eye. Converting between them requires physical display geometry and viewing distance.

## Required information

For degree-based margins provide screen width/height in pixels, physical screen width/height in the same physical unit, and viewing distance in that unit.

```python
deg = ep.convert_aoi_margin_to_degrees(
    (20, 20),
    screen_width_px=1920,
    screen_height_px=1080,
    viewing_distance=60,
    physical_screen_size=(53.1, 29.9),
)
```

Horizontal and vertical degrees-per-pixel are retained separately because screen pixel density need not be identical on both axes.

## Safeguard

A degree-based perturbation fails if neither explicit degrees-per-pixel values nor complete screen/viewing geometry is available. Units are never changed silently.

## Reporting

Report resolution, physical screen dimensions, viewing distance, and perturbation unit. If viewing distance was not controlled, state that limitation explicitly.

Continue to [Interpreting AOI Assignment Stability](assignment-stability.md).