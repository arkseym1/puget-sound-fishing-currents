#!/usr/bin/env python3
"""
Build the Puget Sound Fishing Currents site: pulls NOAA SSCOFS hourly currents
(all 10 model layers) once per hour of the timeline, works out the current at
50, 100 and 150 ft below the surface, and writes compact data for every area,
plus the viewer page itself.

    site/index.html                 the viewer (copied from web/index.html)
    site/data/regions.json          list of areas, depths, colour scale, update time
    site/data/<area>.json           mesh, water level, depth contours
    site/data/<area>-<ft>.json      hourly currents at that depth

How depths work: SSCOFS has 10 terrain-following "sigma" layers - each layer is
a fixed fraction of the local water depth, so no layer sits at a fixed depth.
For each triangle and hour we find how deep each layer's centre is (fraction x
(bottom depth + water level)) and blend linearly between the two layers either
side of the target depth. Above the top layer centre we use the top layer;
below the bottom layer centre, the bottom layer. Triangles whose bottom is
shallower than the target depth are marked as having no water at that depth.

The browser interpolates between the hourly model outputs and animates the
flow, so the page is small and smooth.

    python build_data.py                     # today 5 AM to end of the latest forecast
    python build_data.py --files f1.nc ...   # local SSCOFS files (testing)
"""
import argparse
import base64
import json
import os
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import sscofs_westpoint as S

SITE_REGIONS = [  # (area id in sscofs_westpoint.REGIONS, tab name)
    ("westpoint", "West Point"),
    ("golden-gardens", "Golden Gardens"),
    ("westpoint-goldengardens", "West Point–Golden Gardens"),
    ("richmond-edmonds", "Richmond–Edmonds"),
    ("useless-bay", "Double Bluff–Pt No Point"),
    ("admiralty-inlet", "Admiralty Inlet"),
    ("west-beach", "West Beach"),
    ("deception-pass", "Deception Pass"),
]
DEPTH_FT = [50, 100, 150, 200, 300, 400, 600]   # contour lines drawn on the map
TARGET_FT = [50, 100, 150]                        # depths offered in the viewer
NO_WATER = -32768                                  # int16 marker: seafloor shallower than the target depth
UV_SCALE = 1000          # stored as int16 mm/s
RETRIES = 4              # tries per hourly file before skipping it
RETRY_WAIT = 20          # seconds; grows with each attempt (20, 40, 60)
MAX_SKIP_FRACTION = 0.2  # fail the update (keep the old site) if more than this many hours are lost
HERE = os.path.dirname(os.path.abspath(__file__))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def depth_contours(g):
    fig, ax = plt.subplots()
    tri = mtri.Triangulation(g.lon, g.lat, g.tris)
    levels_m = [f / S.M_TO_FT for f in DEPTH_FT]
    if np.nanmax(g.h) <= levels_m[0]:
        plt.close(fig)
        return []
    cs = ax.tricontour(tri, g.h, levels=levels_m)
    out = []
    for ft, path in zip(DEPTH_FT, cs.get_paths()):
        v, c = path.vertices, path.codes
        if len(v) == 0:
            continue
        starts = list(np.where(c == 1)[0]) if c is not None else [0]
        lines = []
        for a, b in zip(starts, starts[1:] + [len(v)]):
            seg = v[a:b]
            if len(seg) >= 2:
                lines.append(np.round(seg.ravel(), 5).tolist())
        if lines:
            out.append({"ft": ft, "lines": lines})
    plt.close(fig)
    return out


def merged_ranges(spans, gap):
    """Merge (a, b) index spans that overlap or sit within `gap` of each other."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def sigma_centres(ds, g):
    """Layer-centre sigma (negative fraction of depth) for each subset element: (K, ne)."""
    v = ds.variables["siglay"]
    if v.ndim == 1:                                  # same fractions everywhere
        sig = S.arr(v[:]).astype(np.float64)
        return np.repeat(sig[:, None], len(g.nv), axis=1)
    lo, hi = int(g.nodes.min()), int(g.nodes.max()) + 1
    sig = S.arr(v[:, lo:hi]).astype(np.float64)     # (K, nodes in range)
    return sig[:, g.nv - lo].mean(axis=2)           # (K, ne)


def at_depth(u, v, sig, depth_m, target_m):
    """Blend layer velocities (K, ne) to `target_m` below the surface.
    sig: (K, ne) layer-centre sigma; depth_m: (ne,) total water depth (bottom + water level)."""
    K = u.shape[0]
    z = -sig * depth_m[None, :]                      # layer-centre depth below surface, increasing with k
    k = np.clip((z <= target_m).sum(axis=0) - 1, 0, K - 2)
    za, zb = z[k, np.arange(z.shape[1])], z[k + 1, np.arange(z.shape[1])]
    w = np.clip((target_m - za) / np.where(zb - za > 1e-6, zb - za, 1e-6), 0, 1)
    cols = np.arange(u.shape[1])
    uo = u[k, cols] * (1 - w) + u[k + 1, cols] * w
    vo = v[k, cols] * (1 - w) + v[k + 1, cols] * w
    return uo, vo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="site")
    ap.add_argument("--start-hour", type=int, default=5, help="local hour today to start (default 5)")
    ap.add_argument("--source", choices=["auto", "dap", "s3"], default="auto")
    ap.add_argument("--files", nargs="+", help="local SSCOFS fields files instead of NOAA")
    a = ap.parse_args()

    if a.files:
        sources = [(os.path.basename(p), "local", p) for p in a.files]
        run_note = "local files"
    else:
        now = datetime.now(S.LOCAL_TZ)
        start = now.replace(hour=a.start_hour, minute=0, second=0, microsecond=0)
        start = start.astimezone(timezone.utc)
        loc = S.Locator(a.source)
        plan = S.plan_frames(loc, start, 120, 1)        # stops at the forecast's end
        sources = []
        for t, (c, kind, hh) in plan:
            hit = loc.find(c, kind, hh)
            if hit:
                sources.append((S.fname(c, kind, hh), hit[0], hit[1]))
            else:
                print("  missing", S.fname(c, kind, hh))
        if not sources:
            raise SystemExit("No model files available.")
        fc = [s[0] for s in sources if ".fields.f" in s[0]]
        run_note = ("forecast run " + fc[0].split(".")[2] + " " + fc[0].split(".")[1][1:3] + "Z"
                    if fc else "nowcast")

    grids, data, times = {}, {}, []
    ranges, nrange, sig = [], [], {}
    failed = []

    def read_hour(name, kind, url):
        """Read one hourly file into grids/data/times. Raises on any read problem."""
        ds, cleanup = S.open_any(kind, url, False)
        try:
            if not grids:
                # set up the areas once; build into locals so a read failure here
                # leaves nothing half-initialised for the retry
                gs = {}
                for rid, _ in SITE_REGIONS:
                    S.REGION = S.REGIONS[rid]
                    try:
                        g = S.build_grid(ds, S.REGION["box"], margin=0.004)
                    except SystemExit as e:
                        print(f"    {rid}: skipped ({e})")
                        continue
                    gs[rid] = g
                    print(f"    {rid}: {len(g.tris)} triangles, median edge {g.med_edge_m:.0f} m")
                if not gs:
                    raise SystemExit("No areas overlap the model mesh.")
                nodes_all = np.concatenate([g.nodes for g in gs.values()])
                nr = [int(nodes_all.min()), int(nodes_all.max()) + 1]
                sg = {}
                for rid, g in gs.items():
                    sg[rid] = sigma_centres(ds, g)
                    g.h_e = g.h[g.tris].mean(axis=1)            # bottom depth per triangle (m)
                    print(f"    {rid}: {sg[rid].shape[0]} layers, deepest {g.h_e.max() * S.M_TO_FT:.0f} ft")
                # read only the element ranges the areas use (all layers is ~10x the data)
                ranges[:] = merged_ranges([(g.e0, g.e1) for g in gs.values()], 20000)
                nrange[:] = nr
                sig.update(sg)
                for rid, g in gs.items():
                    data[rid] = {"z": [], **{ft: {"u": [], "v": []} for ft in TARGET_FT}}
                grids.update(gs)
                tot = sum(b - a for a, b in ranges)
                print(f"    reading {len(ranges)} element range(s), {tot} elements x all layers per hour")
            # read everything for this hour first, then commit, so a failure
            # part-way through never leaves the areas out of step
            t = S.file_time(ds)
            uv = []
            for a_, b_ in ranges:
                uv.append((a_, S.arr(ds.variables["u"][0, :, a_:b_]).astype(np.float32),
                               S.arr(ds.variables["v"][0, :, a_:b_]).astype(np.float32)))
            n0, n1 = nrange
            zeta = S.arr(ds.variables["zeta"][0, n0:n1]).astype(np.float64)
            zs = {rid: float(S.arr(ds.variables["zeta"][0, g.wp_node:g.wp_node + 1])[0])
                  for rid, g in grids.items()}
        finally:
            try:
                ds.close()
            except Exception:
                pass
            if cleanup and os.path.exists(cleanup):
                os.remove(cleanup)
        for rid, g in grids.items():
            gel = g.e0 + g.eloc                                   # global element ids
            a_, U, V = next(r for r in uv if r[0] <= g.e0 and g.e1 <= r[0] + r[1].shape[1])
            u = np.nan_to_num(U[:, gel - a_]); v = np.nan_to_num(V[:, gel - a_])
            ze = np.nan_to_num(zeta[g.nv - n0]).mean(axis=1)
            depth = np.maximum(g.h_e + ze, 0.1)
            for ft in TARGET_FT:
                tu, tv = at_depth(u, v, sig[rid], depth, ft / S.M_TO_FT)
                dry = g.h_e < ft / S.M_TO_FT                     # seafloor above the target depth
                tu[dry] = np.nan; tv[dry] = np.nan
                data[rid][ft]["u"].append(tu.astype(np.float32))
                data[rid][ft]["v"].append(tv.astype(np.float32))
            data[rid]["z"].append(zs[rid])
        times.append(t)

    for i, (name, kind, url) in enumerate(sources, 1):
        print(f"  [{i}/{len(sources)}] {name}", flush=True)
        for attempt in range(1, RETRIES + 1):
            try:
                read_hour(name, kind, url)
                break
            except SystemExit:
                raise
            except Exception as e:                      # NOAA server hiccup, timeout, etc.
                print(f"    attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}", flush=True)
                if attempt < RETRIES:
                    time.sleep(RETRY_WAIT * attempt)
        else:
            print(f"    giving up on {name}; the viewer will blend across this hour", flush=True)
            failed.append(name)

    if not times:
        raise SystemExit("Couldn't read any model hours - keeping the previously published site.")
    if len(failed) > MAX_SKIP_FRACTION * len(sources):
        raise SystemExit(f"{len(failed)} of {len(sources)} hours failed - keeping the "
                         f"previously published site rather than publishing a patchy one.")
    if failed:
        print(f"  WARNING: skipped {len(failed)} hour(s): {', '.join(failed)}")

    order = np.argsort([t.timestamp() for t in times])
    keep, seen = [], set()
    for k in order:                       # drop duplicate valid times
        ts = int(times[k].timestamp())
        if ts not in seen:
            seen.add(ts)
            keep.append(k)
    epochs = [int(times[k].timestamp()) for k in keep]

    ddir = os.path.join(a.out, "data")
    os.makedirs(ddir, exist_ok=True)
    regions_index = []
    for rid, tab in SITE_REGIONS:
        if rid not in grids:
            continue
        R, g, d = S.REGIONS[rid], grids[rid], data[rid]
        # one colour scale per area for all depths, so colours compare between depths;
        # never let it cap the data: raise it in 0.5 kn steps if needed
        peaks = {}
        for ft in TARGET_FT:
            sp = [np.hypot(d[ft]["u"][k], d[ft]["v"][k]) for k in keep]
            peaks[ft] = float(np.nanmax(sp)) * S.MS_TO_KN if np.isfinite(sp).any() else 0.0
        vmax = max(R["vmax"] * 0.75, np.ceil(max(peaks.values()) * 2) / 2)
        print(f"  {rid}: peaks " + ", ".join(f"{ft} ft {p:.2f} kn" for ft, p in peaks.items())
              + f"; scale 0-{vmax:g} kn")

        def q(arrs):
            out = []
            for k in keep:
                x = np.round(np.nan_to_num(arrs[k], nan=0) * UV_SCALE)
                x = np.clip(x, -32767, 32767).astype("<i2")
                x[np.isnan(arrs[k])] = NO_WATER
                out.append(b64(x))
            return out

        depths = []
        for ft in TARGET_FT:
            wet = int(np.isfinite(d[ft]["u"][keep[0]]).sum())
            if wet == 0:
                print(f"    {rid}: no water as deep as {ft} ft - depth not offered")
                continue
            dpath = os.path.join(ddir, f"{rid}-{ft}.json")
            with open(dpath, "w") as f:
                json.dump({"ft": ft, "uvScale": UV_SCALE, "noWater": NO_WATER,
                           "u": q(d[ft]["u"]), "v": q(d[ft]["v"])}, f, separators=(",", ":"))
            print(f"  wrote {dpath} ({os.path.getsize(dpath) / 1e6:.2f} MB, {wet}/{len(g.tris)} triangles deep enough)")
            depths.append(ft)
        if not depths:
            continue
        out = {
            "id": rid, "title": R["title"], "box": R["box"], "vmax": float(vmax),
            "ref": {"lon": R["ref"][0], "lat": R["ref"][1]},
            "labels": [{"name": n, "lon": x, "lat": y, "kind": kd} for n, x, y, kd in R["labels"]],
            "lon": b64(g.lon.astype("<f4")), "lat": b64(g.lat.astype("<f4")),
            "tris": b64(g.tris.astype("<u4").ravel()),
            "times": epochs, "depths": depths,
            "zetaFt": [round(d["z"][k] * S.M_TO_FT, 2) for k in keep],
            "contours": depth_contours(g),
            "meshMedianM": round(g.med_edge_m),
        }
        path = os.path.join(ddir, rid + ".json")
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
        regions_index.append({"id": rid, "tab": tab, "title": R["title"], "depths": depths})

    cmap = plt.get_cmap("turbo")
    turbo = ["#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(i / 63)[:3]) for i in range(64)]
    stamp = datetime.now(timezone.utc)
    with open(os.path.join(ddir, "regions.json"), "w") as f:
        json.dump({"regions": regions_index, "depths": TARGET_FT, "turbo": turbo, "run": run_note,
                   "updated": int(stamp.timestamp()), "stamp": stamp.strftime("%Y%m%d%H%M")}, f)
    shutil.copy(os.path.join(HERE, "web", "index.html"), os.path.join(a.out, "index.html"))
    print(f"Built {len(regions_index)} areas x {len(TARGET_FT)} depths, {len(epochs)} hours")


if __name__ == "__main__":
    main()
