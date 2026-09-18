# AOI Perturbation Sensitivity Analysis

The sensitivity engine applies declared geometry changes, remaps observations, recomputes AOI features, and optionally reruns the same user-supplied model.

## Supported operations

- dilation and erosion;
- x-only, y-only, and combined x/y translation;
- reproducible uniform jitter;
- anisotropic x/y expansion;
- margins in pixels or degrees;
- rectangles and polygons.

Convex polygons support true geometric offsetting. Concave dilation/erosion is rejected instead of being silently convexified or approximated.

```python
grid = ep.create_aoi_perturbation_grid(
    dilations=[0.25, 0.50, 1.00],
    erosions=[0.25],
    translations_x=[0.50],
    translations_y=[0.50],
    translations_xy=[(0.25, -0.25)],
    unit="deg",
    screen_width_px=1024,
    screen_height_px=768,
    viewing_distance=60,
    physical_screen_size=(53.1, 29.9),
)
```

## Failure behavior

`translations_xy` accepts one or more explicit x/y pairs; NumPy arrays and other iterable inputs are accepted without truth-value coercion. Branches that collapse or invalidate geometry are recorded as failed. Screen-edge behavior is also explicit through `warn`, `clip`, `error`, or `allow`. Jitter retains its seed in the branch specification.

Run the [worked example](../../examples/aoi-perturbation-sensitivity.md) after reading [Pixels versus Degrees of Visual Angle](pixels-versus-degrees.md).