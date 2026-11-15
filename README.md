# Puget Sound Fishing Currents

Animated current maps **at 50, 100 and 150 ft below the surface** for West Point, Golden Gardens,
West Point–Golden Gardens, Richmond Beach–Edmonds, Double Bluff / Maxwelton / Point No Point,
Admiralty Inlet, West Beach and Deception Pass, from NOAA's Salish Sea and Columbia River
Operational Forecast System (SSCOFS), on the model's native triangle mesh.

Built from the Puget Sound Currents (surface) app.

## How the depths work

SSCOFS has 10 terrain-following "sigma" layers: each layer is a fixed fraction of the local
water depth, so no layer sits at a fixed depth. For every triangle and every hour,
`build_data.py` works out how deep each layer's centre is (fraction × (bottom depth + water
level)) and blends linearly between the two layers either side of the target depth. Above the
top layer centre the top layer is used; below the bottom layer centre, the bottom layer.
Where the seafloor is shallower than the chosen depth the map shows grey ("no water at this
depth"), and the chosen depth's contour is highlighted.

In deep water the layers are far apart (in 600 ft of water, about 60 ft), so a value at a
set depth is a blend, not a direct model output.

## Pieces

- `build_data.py` – fetches SSCOFS output (all layers) and writes `site/data/<area>.json`
  (mesh, water level, contours) and `site/data/<area>-<ft>.json` (currents at that depth)
- `web/index.html` – the viewer (canvas, no libraries), with the depth selector
- `sscofs_westpoint.py` – shared SSCOFS access and area definitions (`REGIONS`); its standalone
  still-image/GIF renderer is surface or depth-averaged only
- `.github/workflows/currents.yml` – the schedule (after each of NOAA's four daily runs)

To add an area: add it to `REGIONS` in `sscofs_westpoint.py` and to `SITE_REGIONS` in
`build_data.py`. To change the depths: edit `TARGET_FT` in `build_data.py`.

Model guidance only – not for navigation.
