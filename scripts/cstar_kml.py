# -*- coding: utf-8 -*-
"""
C-Star Operational KMZ: ERDDAP → KMZ with NHC Wallets → Cone → Outlook
- 48h default lookback for tracks + charts.
- Clickable track dots with per-time observations (SI + English).
- Alert banner + red "alert ring" around platforms inside cone/outlook (single banner even if both hit).
- Prompt for cone URL (KMZ/KML/GeoJSON). If blank, scan NHC Atlantic Wallets (nhc_at1..nhc_at5).
- If no active cone, skip cone search and query NHC Outlook polygons directly.
- Charts embedded at KMZ root for maximum viewer compatibility.
- Icons embedded from local ./icons/<size>/pc#.png (fallback 1x1 if missing).
- Titles are left-aligned; small right-aligned "Max/Dominant" notes share the title row (no box).
- Vertical spacing tightened; footer has its own short row.
"""

import argparse, csv, html, io, os, re, sys, zipfile, math, json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import requests

# Optional plot + geometry helpers
try:
    from matplotlib.path import Path as MplPath  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, AutoDateLocator
    HAVE_MPL = True
except Exception:
    MplPath = None
    HAVE_MPL = False

try:
    from shapely.geometry import Point as ShpPoint, Polygon as ShpPolygon  # type: ignore
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

# ---------- Defaults ----------
DEFAULT_SERVERS = [
    "https://data.pmel.noaa.gov/pmel/erddap",
    "https://www.aoml.noaa.gov/erddap",
]
NHC_WALLET_URLS = [
    "https://www.nhc.noaa.gov/nhc_at1.xml",
    "https://www.nhc.noaa.gov/nhc_at2.xml",
    "https://www.nhc.noaa.gov/nhc_at3.xml",
    "https://www.nhc.noaa.gov/nhc_at4.xml",
    "https://www.nhc.noaa.gov/nhc_at5.xml",
]
NHC_CONE_INDEX = "https://www.nhc.noaa.gov/storm_graphics/api/"
NHC_OUTLOOK_LAYER = "https://mapservices.weather.noaa.gov/tropical/rest/services/tropical/NHC_tropical_weather/MapServer/3"

ALERT_TEXT = "Alert-This platform is located in or near an official NHC outlook area or NHC tropical cyclone forecast cone."

# Track line colors (KML aabbggrr)
LINE_COLORS = {
    "PC2": "ff0000ff",  # red
    "PC3": "ffff0000",  # blue
    "PC4": "ff00ff00",  # green
    "PC5": "ffffff00",  # cyan
    "PC6": "ff00ffff",  # yellow
    "PC8": "ffff00ff",  # magenta
}

DARK_BLUE = "#003366"  # annotation text color

def log(s):  print(s, flush=True)
def ok(s):  print(f"✓ {s}", flush=True)
def warn(s):print(f"⚠️  {s}", flush=True)
def err(s): print(f"❌ {s}", flush=True)

# ---------- Time helpers ----------
def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None

def parse_iso_z(s: str) -> Optional[datetime]:
    if not s: return None
    try:
        if s.endswith("Z"): s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None

def utc_to_et_str(iso_z: str) -> str:
    dt = parse_iso_z(iso_z)
    if not dt or _ET is None: return ""
    try:
        return dt.astimezone(_ET).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return ""

# ---------- HTTP ----------
def http_get(url: str, timeout: int = 20) -> requests.Response:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r

# ---------- ERDDAP ----------
def probe_server(server: str) -> None:
    url = f"{server}/search/index.html?searchFor=hello"
    r = http_get(url)
    if r.status_code != 200:
        raise RuntimeError(f"probe got HTTP {r.status_code}")

def erddap_search_csv(server: str, query: str) -> List[Dict[str, str]]:
    url = f"{server}/search/index.csv?page=1&itemsPerPage=2000&searchFor={requests.utils.quote(query)}"
    r = http_get(url)
    return list(csv.DictReader(io.StringIO(r.content.decode("utf-8", errors="replace"))))

def info_json(server: str, dataset_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    url = f"{server}/info/{dataset_id}/index.json"
    js = http_get(url).json()
    table = js.get("table", {})
    names = table.get("columnNames") or []
    rows  = table.get("rows") or []
    if not rows or not names: return None
    def col(alts):
        for a in alts:
            if a in names: return names.index(a)
        return None
    i_var   = col(["variableName","variable"])
    i_attr  = col(["attributeName","attribute"])
    i_value = col(["value"])
    i_dtype = col(["dataType"])
    if i_var is None or i_attr is None or i_value is None: return None
    meta: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        var, attr, val = str(r[i_var]), str(r[i_attr]), r[i_value]
        meta.setdefault(var, {})
        if i_dtype is not None and r[i_dtype]:
            meta[var]["dataType"] = str(r[i_dtype]).lower()
        if attr in ("long_name","standard_name","units","axis","dataType"):
            if attr == "dataType": meta[var]["dataType"] = str(val).lower()
            else: meta[var][attr] = val
    for v,m in meta.items():
        if "long_name" not in m and "standard_name" in m:
            m["long_name"] = m["standard_name"]
        m.setdefault("units","")
    return meta

def info_csv(server: str, dataset_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    url = f"{server}/info/{dataset_id}/index.csv"
    r = http_get(url)
    meta: Dict[str, Dict[str, Any]] = {}
    have_any = False
    for row in csv.DictReader(io.StringIO(r.content.decode("utf-8", errors="replace"))):
        var = row.get("Variable Name") or row.get("Variable") or row.get("variableName") or row.get("variable")
        attr = row.get("Attribute Name") or row.get("Attribute") or row.get("attributeName") or row.get("attribute")
        val = row.get("Value") or row.get("value")
        dtype = row.get("Data Type") or row.get("dataType")
        if not var or not attr: continue
        have_any = True
        meta.setdefault(var, {})
        if dtype: meta[var]["dataType"] = str(dtype).lower()
        if attr in ("long_name","standard_name","units","axis","dataType"):
            if attr == "dataType": meta[var]["dataType"] = str(val).lower()
            else: meta[var][attr] = val
    if not have_any: return None
    for v,m in meta.items():
        if "long_name" not in m and "standard_name" in m:
            m["long_name"] = m["standard_name"]
        m.setdefault("units","")
    return meta

def info_csvp(server: str, dataset_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
    url = f"{server}/tabledap/{dataset_id}.csvp"
    r = http_get(url)
    text = r.content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header_ix = None
    for i, line in enumerate(lines):
        if "Variable Name" in line or "Variable" in line:
            header_ix = i; break
    if header_ix is None: return None
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_ix:])))
    meta: Dict[str, Dict[str, Any]] = {}
    for row in reader:
        var = row.get("Variable Name") or row.get("Variable")
        if not var: continue
        dt  = (row.get("Data Type") or "").lower()
        ln  = row.get("Long Name") or row.get("long_name") or ""
        units = row.get("Units") or ""
        meta[var] = {"dataType": dt, "long_name": ln or var, "units": units}
    return meta if meta else None

def fetch_columns_rows(server: str, dataset_id: str, cols: List[str], after_iso: Optional[str], order: str):
    col_list = ",".join(cols)
    base = f"{server}/tabledap/{dataset_id}.json?{col_list}"
    if after_iso: base += f"&time>={after_iso}"
    if order:     base += f"&{order}"
    js = http_get(base).json()
    table = js.get("table", {})
    return (table.get("columnNames") or [], table.get("rows") or [])

def fetch_latest_position(server: str, dataset_id: str, id_field: Optional[str], wmo_field: Optional[str]):
    want = ["time","latitude","longitude"]
    if id_field:  want.append(id_field)
    if wmo_field: want.append(wmo_field)
    names, rows = fetch_columns_rows(server, dataset_id, want, None, 'orderByMax("time")')
    if not rows: return None
    return dict(zip(names, rows[0]))

def fetch_track(server: str, dataset_id: str, hours: int) -> List[Tuple[str,float,float]]:
    names, rows = fetch_columns_rows(server, dataset_id, ["time","latitude","longitude"],
                                     iso_utc(datetime.now(timezone.utc)-timedelta(hours=hours)),
                                     'orderBy("time")')
    if not rows: return []
    i_t, i_la, i_lo = names.index("time"), names.index("latitude"), names.index("longitude")
    out = []
    for r in rows:
        try: out.append((str(r[i_t]), float(r[i_la]), float(r[i_lo])))
        except Exception: pass
    return out

def fetch_vars_series(server: str, dataset_id: str, varnames: List[str], hours: int):
    cols = ["time"] + varnames
    names, rows = fetch_columns_rows(server, dataset_id, cols,
                                     iso_utc(datetime.now(timezone.utc)-timedelta(hours=hours)),
                                     'orderBy("time")')
    if not rows: return names, []
    return names, rows

def fetch_latest_obs_vars(server: str, dataset_id: str, varnames: List[str]) -> Dict[str, Any]:
    cols = ["time"] + varnames
    names, rows = fetch_columns_rows(server, dataset_id, cols, None, 'orderByMax("time")')
    if not rows or not names: return {}
    return dict(zip(names, rows[0]))

# ---------- Geometry helpers ----------
def meters_to_deg(m: float) -> float: return float(m) / 111320.0
def normalize_lon(lon: float) -> float:
    if lon > 180.0:   lon -= 360.0
    elif lon < -180.0:lon += 360.0
    return lon

def even_odd_inside(x: float, y: float, poly: List[Tuple[float, float]], eps_deg: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if abs(y2 - y1) < 1e-15: continue
        if ((y1 > y) != (y2 > y)):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin + eps_deg: inside = not inside
    return inside

def winding_number_inside(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    wn = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if y1 <= y:
            if y2 > y and ((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)) > 0:
                wn += 1
        else:
            if y2 <= y and ((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)) < 0:
                wn -= 1
    return wn != 0

def inside_polygon_robust(lon: float, lat: float, poly: List[Tuple[float, float]], eps_deg: float) -> bool:
    if HAVE_SHAPELY:
        try:
            shp = ShpPolygon(poly); pt  = ShpPoint(lon, lat)
            if shp.contains(pt): return True
            if shp.exterior.distance(pt) <= eps_deg: return True
        except Exception: pass
    if MplPath is not None:
        try:
            p = MplPath(poly, closed=True)
            if p.contains_point((lon, lat), radius=eps_deg): return True
        except Exception: pass
    if even_odd_inside(lon, lat, poly, eps_deg): return True
    if winding_number_inside(lon, lat, poly):    return True
    return False

def point_in_any_robust(lon: float, lat: float, polys: List[List[Tuple[float,float]]], eps_deg: float) -> bool:
    for poly in polys:
        if inside_polygon_robust(lon, lat, poly, eps_deg): return True
    return False

def convex_hull(points: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    pts = sorted(set(points))
    if len(pts) <= 2: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if hull and hull[0] != hull[-1]: hull.append(hull[0])
    return hull

# ---------- Cone parsing / loading ----------
def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def _parse_coords_text(coords_text: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for token in re.split(r"[\s\n\r]+", (coords_text or "").strip()):
        if not token: continue
        parts = token.split(",")
        try:
            lon = float(parts[0]); lat = float(parts[1])
            pts.append((lon, lat))
        except Exception:
            continue
    if len({(round(x,6), round(y,6)) for x,y in pts}) < 3:
        return []
    return pts

def parse_kml_polygons(kml_bytes: bytes) -> List[List[Tuple[float, float]]]:
    import xml.etree.ElementTree as ET
    polys: List[List[Tuple[float, float]]] = []
    try:
        root = ET.fromstring(kml_bytes)
    except Exception:
        return polys

    def rings_under(el):
        out = []
        for child in el.iter():
            if _local(child.tag) == "LinearRing":
                for C in child.iter():
                    if _local(C.tag) == "coordinates":
                        pts = _parse_coords_text(C.text or "")
                        if len(pts) >= 3:
                            out.append(pts)
        return out

    outer = []
    for poly in root.iter():
        if _local(poly.tag) == "Polygon":
            for ob in poly.iter():
                if _local(ob.tag) == "outerBoundaryIs":
                    outer.extend(rings_under(ob))
    if outer: return outer

    any_lr = []
    for poly in root.iter():
        if _local(poly.tag) == "Polygon":
            any_lr.extend(rings_under(poly))
    if any_lr: return any_lr

    for el in root.iter():
        if _local(el.tag) == "LineString":
            for C in el.iter():
                if _local(C.tag) == "coordinates":
                    pts = _parse_coords_text(C.text or "")
                    if len(pts) >= 4 and abs(pts[0][0]-pts[-1][0]) < 1e-6 and abs(pts[0][1]-pts[-1][1]) < 1e-6:
                        polys.append(pts)
    return polys

def load_cone_polys_from_path_or_url(path_or_url: str) -> List[List[Tuple[float, float]]]:
    try:
        if re.match(r'^https?://', path_or_url, re.I):
            rr = http_get(path_or_url, timeout=20)
            data = rr.content
        else:
            with open(path_or_url, "rb") as f:
                data = f.read()
        low = path_or_url.lower()
        if low.endswith(".kmz"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                kmls = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kmls:
                    warn("KMZ had no KML member"); return []
                def score(name: str) -> int:
                    n = name.lower()
                    return (100 if "cone" in n else 0) + (20 if "forecast" in n or "poly" in n else 0)
                best = sorted(kmls, key=score, reverse=True)[0]
                kml_bytes = zf.read(best)
            return parse_kml_polygons(kml_bytes)
        elif low.endswith(".kml"):
            return parse_kml_polygons(data)
        elif low.endswith(".geojson") or low.endswith(".json"):
            gj = json.loads(data.decode("utf-8", errors="replace"))
            polys: List[List[Tuple[float, float]]] = []
            def add_coords(coords):
                if not coords: return
                if isinstance(coords[0][0][0], (float, int)):
                    ring = coords[0]
                    polys.append([(float(x), float(y)) for (x,y) in ring])
                else:
                    for poly in coords:
                        ring = poly[0]
                        polys.append([(float(x), float(y)) for (x,y) in ring])
            if gj.get("type") == "FeatureCollection":
                for feat in gj.get("features", []):
                    geom = feat.get("geometry") or {}
                    t = geom.get("type"); coords = geom.get("coordinates")
                    if t == "Polygon": add_coords(coords)
                    elif t == "MultiPolygon":
                        for poly in coords: add_coords(poly)
            elif gj.get("type") == "Polygon":
                add_coords(gj.get("coordinates"))
            elif gj.get("type") == "MultiPolygon":
                for poly in gj.get("coordinates"):
                    add_coords(poly)
            return polys
        else:
            warn(f"Unsupported cone file type: {path_or_url}")
            return []
    except Exception as e:
        warn(f"Failed to load cone file: {e}")
        return []

# ---------- Wallets → AL-IDs → Cone discovery ----------
def extract_alids(text: str) -> List[str]:
    return sorted(set(re.findall(r'AL\d{2}\d{4}', text, re.I)))

def wallets_active_alids() -> List[str]:
    alids: Set[str] = set()
    for url in NHC_WALLET_URLS:
        try:
            r = http_get(url, timeout=10)
            ids = extract_alids(r.text)
            if ids:
                ok(f"Wallet {url} → AL IDs: {', '.join(ids)}")
                alids.update([s.upper() for s in ids])
            else:
                log(f"Wallet {url} → no active AL IDs.")
        except Exception as e:
            warn(f"Wallet fetch failed ({url}): {e}")
    return sorted(alids)

def latest_cone_url_for_alids(alids: List[str]) -> Optional[str]:
    try:
        idx = http_get(NHC_CONE_INDEX, timeout=20).text
    except Exception as e:
        warn(f"Cone index fetch failed: {e}")
        return None
    best = None; best_adv = -1
    for al in alids:
        pattern = rf'href="({al}_\d+adv_CONE\.kmz)"'
        for m in re.finditer(pattern, idx, re.I):
            href = m.group(1)
            m2 = re.search(r'_(\d+)adv_CONE\.kmz', href, re.I)
            adv = int(m2.group(1)) if m2 else 0
            if adv > best_adv:
                best_adv = adv
                best = href if href.lower().startswith("http") else (NHC_CONE_INDEX + href)
    if best:
        ok(f"Selected latest cone for {', '.join(alids)}: {best} (adv {best_adv:03d})")
    else:
        ok("No *_CONE.kmz matched the wallet AL-IDs.")
    return best

# ---------- NHC Outlook ----------
def fetch_nhc_outlook_polygons() -> List[List[Tuple[float, float]]]:
    url = f"{NHC_OUTLOOK_LAYER}/query"
    params = {"where":"1=1","outFields":"*","returnGeometry":"true","f":"pjson","outSR":"4326"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        feats = js.get("features", [])
        polys: List[List[Tuple[float, float]]] = []
        for ftr in feats:
            geom = ftr.get("geometry") or {}
            rings = geom.get("rings") or []
            for ring in rings:
                pts: List[Tuple[float, float]] = []
                for xy in ring:
                    try:
                        lon, lat = float(xy[0]), float(xy[1])
                        pts.append((lon, lat))
                    except Exception:
                        continue
                if len(pts) >= 3:
                    polys.append(pts)
        if polys:
            ok(f"Fetched {len(polys)} outlook polygon ring(s).")
        else:
            ok("No outlook polygons returned.")
        return polys
    except Exception as e:
        warn(f"Outlook fetch failed: {e}")
        return []

# ---------- Unit helpers ----------
def kt_to_mph(kt: float) -> float: return kt * 1.15078
def c_to_f(c: float) -> float:    return c * 9.0/5.0 + 32.0
def hpa_to_mb(h: float) -> float: return h
def esc(s: Any) -> str: return html.escape("" if s is None else str(s))
def fmt_number(x: Any, nd: int = 2) -> str:
    try:
        xf = float(x)
        if math.isfinite(xf): return f"{xf:.{nd}f}"
    except Exception:
        pass
    return str(x)

# ---------- Balloon content ----------
def fmt_obs_table_from_row(row: Dict[str, Any]) -> str:
    def maybe_float(v):
        try: return float(v)
        except Exception: return float("nan")
    ws_mean = maybe_float(row.get("wind_speed_mean"))
    ws_max  = maybe_float(row.get("wind_speed_max"))
    wdir    = maybe_float(row.get("wind_from_direction_mean"))
    at_c    = maybe_float(row.get("air_temperature_mean"))
    rh      = maybe_float(row.get("relative_humidity_mean"))
    sst_c   = maybe_float(row.get("sea_surface_temperature_mean"))
    pres_h  = maybe_float(row.get("air_pressure_mean"))
    avg_s   = maybe_float(row.get("met_averaging_period"))
    # Conversions
    ws_mean_mph = ws_mean * 1.15078 if math.isfinite(ws_mean) else float("nan")
    ws_max_mph  = ws_max  * 1.15078 if math.isfinite(ws_max)  else float("nan")
    at_f   = c_to_f(at_c)  if math.isfinite(at_c)  else float("nan")
    sst_f  = c_to_f(sst_c) if math.isfinite(sst_c) else float("nan")
    pres_mb = pres_h       if math.isfinite(pres_h) else float("nan")
    rh_pct = rh * 100.0 if math.isfinite(rh) and rh <= 1.5 else rh
    def row_html(label, si_val, si_unit, en_val=None, en_unit=""):
        if en_val is None or not math.isfinite(en_val):
            return f"<tr><td>{esc(label)}</td><td>{fmt_number(si_val)} {esc(si_unit)}</td><td></td></tr>"
        return f"<tr><td>{esc(label)}</td><td>{fmt_number(si_val)} {esc(si_unit)}</td><td>/ {fmt_number(en_val)} {esc(en_unit)}</td></tr>"
    rows = [
        row_html("Wind Speed (mean)", ws_mean, "kt", ws_mean_mph, "mph"),
        row_html("Wind Speed (max)",  ws_max,  "kt", ws_max_mph,  "mph"),
        row_html("Wind From Dir (mean)", wdir, "deg"),
        row_html("Air Temperature (mean)", at_c, "°C", at_f, "°F"),
        row_html("Relative Humidity (mean)", rh_pct, "%"),
        row_html("Sea Surface Temp (mean)", sst_c, "°C", sst_f, "°F"),
        row_html("Air Pressure (mean)", pres_mb, "mb"),
        row_html("Met Averaging Period", avg_s, "s"),
    ]
    return ('<table cellspacing="0" cellpadding="3" style="border-collapse:collapse;font-size:12px">'
            "<thead><tr><th align='left'>Observation</th><th align='left'>SI</th><th align='left'>English</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>")

def balloon_html_latest(label: str, when_iso: str, lat: float, lon: float, wmo: str,
                        show_alert: bool, obs_html: str, chart_rel_path: Optional[str]) -> str:
    et = utc_to_et_str(when_iso) if when_iso else ""
    nhc_html = ""
    if show_alert:
        nhc_html = (
            "<div style=\"padding:8px 10px;margin:0 0 8px 0;"
            "background:#ffe5e5;border:2px solid #cc0000;border-radius:4px;"
            "color:#b00000;font-weight:700;\">"
            f"{esc(ALERT_TEXT)}"
            "</div>"
        )
    chart_tag = f'<div style="margin:6px 0;"><img src="{esc(chart_rel_path)}" width="900"/></div>' if chart_rel_path else ""
    return f"""
<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.35;font-size:13px">
  {nhc_html}
  <div style="margin-bottom:6px">
    <b>{esc(label)}</b> — C-Star USV<br/>
    <b>UTC:</b> {esc(when_iso)}{(' / <b>ET:</b> ' + esc(et)) if et else ''} &nbsp;|&nbsp;
    <b>Lat/Lon:</b> {lat:.4f}, {lon:.4f}{(' &nbsp;|&nbsp; <b>WMO:</b> '+esc(wmo)) if wmo else ''}
  </div>
  <div style="margin:6px 0 8px 0">
    <b>Observations (latest available):</b><br/>{obs_html}
  </div>
  {chart_tag}
  <div style="margin-top:8px;font-size:12px;color:#666">
    <i>Powered by GeoCollaborate®</i>
  </div>
</div>
""".strip()

def balloon_html_dot(label: str, when_iso: str, obs_html: str) -> str:
    et = utc_to_et_str(when_iso) if when_iso else ""
    return f"""
<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.35;font-size:13px">
  <b>{esc(label)}</b> — Observation @ <b>UTC:</b> {esc(when_iso)}{(' / <b>ET:</b> ' + esc(et)) if et else ''}<br/>
  {obs_html}
  <div style="margin-top:6px;font-size:12px;color:#666"><i>Powered by GeoCollaborate®</i></div>
</div>
""".strip()

# ---------- Cardinal / “Dominant” helpers ----------
CARD16 = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
def cardinal_from_deg(deg: float) -> str:
    if not math.isfinite(deg): return ""
    idx = int((deg/22.5)+0.5) % 16
    return CARD16[idx]

def dominant_direction_deg(deg_values: List[float]) -> Optional[float]:
    vals = [(d % 360.0) for d in deg_values if d is not None and math.isfinite(d)]
    if not vals: return None
    # Mode of 10° bins
    counts = [0]*36
    for d in vals:
        counts[int(d//10) % 36] += 1
    k = max(range(36), key=lambda i: counts[i])
    dom = (k*10 + 5) % 360.0  # bin center
    return dom

# ---------- Charts (one PNG per platform) ----------
def _title_right_note(ax, text: str):
    ax.text(0.99, 1.01, text, transform=ax.transAxes, ha="right", va="bottom",
            color=DARK_BLUE, fontsize=8.5, fontweight="bold", clip_on=False)

def build_composite_chart(server: str, did: str, label: str, hours: int,
                          disk_dir: str, kmz_rel_name: str,
                          width_px=900, panel_height_px=160, dpi=110) -> Optional[str]:
    if not HAVE_MPL:
        warn("matplotlib not available; no charts will be embedded.")
        return None
    vars_needed = [
        "sea_surface_temperature_mean",
        "air_temperature_mean",
        "wind_speed_max",
        "relative_humidity_mean",
        "air_pressure_mean",
        "wind_from_direction_mean",
    ]
    try:
        names, rows = fetch_vars_series(server, did, vars_needed, hours)
    except Exception as e:
        warn(f"{label}: chart series fetch failed: {e}")
        return None
    if not rows or not names: return None

    idx = {n: i for i, n in enumerate(names)}
    t_ix = idx.get("time")
    times: List[datetime] = []
    series: Dict[str, List[float]] = {v: [] for v in vars_needed}

    for r in rows:
        try:
            t = parse_iso_z(str(r[t_ix]))
            if not t: continue
            times.append(t)
            for v in vars_needed:
                vi = idx.get(v)
                val = float(r[vi]) if (vi is not None and r[vi] not in (None, "")) else float("nan")
                series[v].append(val)
        except Exception:
            continue

    if not times:
        return None

    rows_panels = 6
    rows_total = rows_panels + 1  # +1 footer row
    fig_h_px = panel_height_px * rows_total
    fig = plt.figure(figsize=(width_px/dpi, fig_h_px/dpi), dpi=dpi, constrained_layout=True)

    # Tighter spacing; footer row slim; slight pads so titles aren't clipped
    gs = fig.add_gridspec(rows_total, 1, height_ratios=[1,1,1,1,1,1,0.36])
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.05, hspace=0.22, wspace=0.05)

    plt.rcParams.update({
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
    })
    date_locator = AutoDateLocator()
    date_fmt = DateFormatter("%m-%d\n%H:%MZ")

    def add_temp_panel(slot, data_c, title, ylabel_left):
        ax = fig.add_subplot(gs[slot, 0])
        ax.plot(times, data_c, linewidth=1.6)
        ax.set_ylabel(ylabel_left)
        ax.set_title(f"{label} — {title} (last {hours}h)", pad=2, loc="left")
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_fmt)
        finite = [v for v in data_c if math.isfinite(v)]
        if finite:
            cmin, cmax = min(finite), max(finite)
            if cmin == cmax: cmin -= 1; cmax += 1
            ax.set_ylim(cmin, cmax)
            ax2 = ax.twinx()
            ax2.set_ylim(c_to_f(cmin), c_to_f(cmax))
            ax2.set_ylabel("degF", rotation=-90, labelpad=10)
        return ax

    # 1) SST (tight label)
    ax1 = add_temp_panel(0, series["sea_surface_temperature_mean"], "Sea Surface Temperature", "SST (°C)")
    sst_vals = series["sea_surface_temperature_mean"]
    finite_sst = [(i, v) for i, v in enumerate(sst_vals) if math.isfinite(v)]
    if finite_sst:
        idx_max, vmax = max(finite_sst, key=lambda t: t[1])
        t_max_str = times[idx_max].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        _title_right_note(ax1, f"Max = {vmax:.2f} °C ({c_to_f(vmax):.1f} °F) @ {t_max_str}")

    # 2) Air Temp
    ax2 = add_temp_panel(1, series["air_temperature_mean"], "Air Temperature", "Air Temp (°C)")
    at_vals = series["air_temperature_mean"]
    finite_at = [(i, v) for i, v in enumerate(at_vals) if math.isfinite(v)]
    if finite_at:
        idx_max, vmax = max(finite_at, key=lambda t: t[1])
        t_max_str = times[idx_max].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        _title_right_note(ax2, f"Max = {vmax:.2f} °C ({c_to_f(vmax):.1f} °F) @ {t_max_str}")

    # 3) Wind Speed Max (kt)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(times, series["wind_speed_max"], linewidth=1.6)
    ax3.set_ylabel("Wind Max (kt)")
    ax3.set_title(f"{label} — Wind Speed Max (last {hours}h)", pad=2, loc="left")
    ax3.xaxis.set_major_locator(date_locator); ax3.xaxis.set_major_formatter(date_fmt)
    ws_vals = series["wind_speed_max"]
    finite_ws = [(i, v) for i, v in enumerate(ws_vals) if math.isfinite(v)]
    if finite_ws:
        idx_max, vmax = max(finite_ws, key=lambda t: t[1])
        t_max_str = times[idx_max].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        _title_right_note(ax3, f"Max = {vmax:.2f} kt ({kt_to_mph(vmax):.1f} mph) @ {t_max_str}")

    # 4) Relative Humidity (%)
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(times, series["relative_humidity_mean"], linewidth=1.6)
    ax4.set_ylabel("RH (%)")
    ax4.set_title(f"{label} — Relative Humidity (last {hours}h)", pad=2, loc="left")
    ax4.xaxis.set_major_locator(date_locator); ax4.xaxis.set_major_formatter(date_fmt)
    rh_vals = series["relative_humidity_mean"]
    finite_rh = [(i, v) for i, v in enumerate(rh_vals) if math.isfinite(v)]
    if finite_rh:
        idx_max, vmax = max(finite_rh, key=lambda t: t[1])
        t_max_str = times[idx_max].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        _title_right_note(ax4, f"Max = {vmax:.2f} % @ {t_max_str}")

    # 5) Air Pressure (mb)
    ax5 = fig.add_subplot(gs[4, 0])
    press_mb = [hpa_to_mb(v) if math.isfinite(v) else float("nan") for v in series["air_pressure_mean"]]
    ax5.plot(times, press_mb, linewidth=1.6)
    ax5.set_ylabel("Pressure (mb)")
    ax5.set_title(f"{label} — Air Pressure (last {hours}h)", pad=2, loc="left")
    ax5.xaxis.set_major_locator(date_locator); ax5.xaxis.set_major_formatter(date_fmt)
    finite_p = [(i, v) for i, v in enumerate(press_mb) if math.isfinite(v)]
    if finite_p:
        idx_max, vmax = max(finite_p, key=lambda t: t[1])
        t_max_str = times[idx_max].astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        _title_right_note(ax5, f"Max = {vmax:.2f} mb @ {t_max_str}")

    # 6) Wind Direction (deg)
    ax6 = fig.add_subplot(gs[5, 0])
    ax6.plot(times, series["wind_from_direction_mean"], linewidth=1.3)
    ax6.set_ylabel("Wind Dir (°)")
    ax6.set_title(f"{label} — Wind Direction (last {hours}h)", pad=2, loc="left")
    ax6.set_ylim(0, 360)
    ax6.xaxis.set_major_locator(date_locator); ax6.xaxis.set_major_formatter(date_fmt)
    degs = series["wind_from_direction_mean"]
    vals = [v for v in degs if math.isfinite(v)]
    if vals:
        dom = dominant_direction_deg(vals)
        if dom is not None:
            card = cardinal_from_deg(dom)
            _title_right_note(ax6, f"Dominant = {dom:.0f}° ({card})")

    # 7) Footer row (short)
    ax_footer = fig.add_subplot(gs[6, 0])
    ax_footer.axis("off")
    ax_footer.text(0.01, 0.5, "Powered by GeoCollaborate®", fontsize=9, ha="left", va="center", color="#333333")

    os.makedirs(disk_dir, exist_ok=True)
    disk_path = os.path.join(disk_dir, kmz_rel_name)
    fig.savefig(disk_path, format="png", dpi=dpi)
    plt.close(fig)
    return kmz_rel_name

# ---------- Polygon normalization ----------
def normalize_polygons_longitudes(polys: List[List[Tuple[float, float]]]) -> Tuple[List[List[Tuple[float,float]]], str]:
    if not polys: return polys, "no polys"
    def norm(x):
        if x > 180.0:   return x - 360.0
        if x < -180.0:  return x + 360.0
        return x
    lons = [x for poly in polys for (x, _) in poly]
    if not lons: return polys, "no lons"
    note = "already [-180,180]"
    polys = [[(norm(x), y) for (x, y) in poly] for poly in polys]
    return polys, note

# ---------- KML builders ----------
def kml_header(doc_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document>
  <name>{esc(doc_name)}</name>
  <open>1</open>
"""

def kml_footer() -> str:
    return "</Document>\n</kml>\n"

def style_block_icon(pid: str, icon_href_kmz: str, alert_tint: bool=False) -> str:
    color = "<color>ff0000ff</color>" if alert_tint else ""
    return f"""  <Style id="{pid}{'_alert' if alert_tint else ''}_icon">
    <IconStyle>{color}<scale>1.25</scale><Icon><href>{esc(icon_href_kmz)}</href></Icon></IconStyle>
    <LabelStyle><scale>1.1</scale></LabelStyle>
  </Style>
"""

def style_block_line(pid: str, line_color: str) -> str:
    return f"""  <Style id="{pid}_line">
    <LineStyle><color>{line_color}</color><width>3.0</width></LineStyle>
  </Style>
"""

def style_block_dot(pid: str) -> str:
    return f"""  <Style id="{pid}_dot">
    <IconStyle><scale>0.8</scale><Icon><href>dot.png</href></Icon></IconStyle>
  </Style>
"""

def style_block_alert_ring(pid: str) -> str:
    return f"""  <Style id="{pid}_ring">
    <LineStyle><color>ff0000ff</color><width>2.5</width></LineStyle>
    <PolyStyle><color>660000ff</color></PolyStyle>
  </Style>
"""

def placemark_track(pid: str, label: str, line_color: str, track: List[Tuple[str,float,float]], hours: int) -> str:
    if len(track) < 2: return ""
    coords = " ".join([f"{lon:.6f},{lat:.6f},0" for (t, lat, lon) in track])
    t0, t1 = track[0][0], track[-1][0]
    desc = f"{label} recent track — last {hours}h<br/>From: {t0} / {utc_to_et_str(t0)}<br/>To: {t1} / {utc_to_et_str(t1)}"
    return f"""    <Placemark>
      <name>{esc(label)} Track (last {hours}h)</name>
      <styleUrl>#{pid}_line</styleUrl>
      <description><![CDATA[{esc(desc)}]]></description>
      <LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>
    </Placemark>
"""

def placemark_point(pid: str, name: str, when: str, html_desc: str, lat: float, lon: float, style_suffix: str = "_icon") -> str:
    return f"""    <Placemark>
      <name>{esc(name)}</name>
      <styleUrl>#{pid}{style_suffix}</styleUrl>
      <TimeStamp><when>{esc(when)}</when></TimeStamp>
      <description><![CDATA[{html_desc}]]></description>
      <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
    </Placemark>
"""

def placemark_ring(pid: str, lat: float, lon: float, radius_m: float = 12000.0) -> str:
    n = 72
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * max(0.0001, math.cos(math.radians(lat)))
    dlat = radius_m / m_per_deg_lat
    dlon = radius_m / m_per_deg_lon
    pts = []
    for i in range(n+1):
        ang = 2*math.pi*i/n
        pts.append((lon + dlon*math.cos(ang), lat + dlat*math.sin(ang)))
    coords = " ".join([f"{x:.6f},{y:.6f},0" for (x,y) in pts])
    return f"""    <Placemark>
      <name>{esc(pid)} Alert Ring</name>
      <styleUrl>#{pid}_ring</styleUrl>
      <Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
    </Placemark>
"""

def placemark_track_dot(pid: str, label: str, when: str, lat: float, lon: float,
                        html_desc: str, region_deg: float, lod_min: int) -> str:
    north = lat + region_deg; south = lat - region_deg
    east  = lon + region_deg; west  = lon - region_deg
    return f"""    <Placemark>
      <name>{esc(label)} · {esc(when)}</name>
      <styleUrl>#{pid}_dot</styleUrl>
      <TimeStamp><when>{esc(when)}</when></TimeStamp>
      <description><![CDATA[{html_desc}]]></description>
      <Region>
        <LatLonAltBox><north>{north:.6f}</north><south>{south:.6f}</south><east>{east:.6f}</east><west>{west:.6f}</west></LatLonAltBox>
        <Lod><minLodPixels>{lod_min}</minLodPixels></Lod>
      </Region>
      <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
    </Placemark>
"""

def kmz_icon_rel(size_str: str, label: str) -> str:
    return f"icons/{size_str}/{label.lower()}.png"

def disk_icon_path(base_dir: str, size_str: str, label: str) -> str:
    return os.path.join(base_dir, size_str, f"{label.lower()}.png")

# ---------- Helpers for dot obs lookups ----------
def nearest_row_for_time(names: List[str], rows: List[List[Any]], t_iso: str, max_tol_sec: int = 900) -> Optional[Dict[str, Any]]:
    if not rows: return None
    t_ix = names.index("time")
    target = parse_iso_z(t_iso)
    if not target: return None
    best = None; best_dt = 1e18
    for r in rows:
        rt = parse_iso_z(str(r[t_ix]))
        if not rt: continue
        dt = abs((rt - target).total_seconds())
        if dt < best_dt:
            best_dt = dt; best = r
    if best is None or best_dt > max_tol_sec:
        return None
    return dict(zip(names, best))

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(description="C-Star Operational KMZ with clickable track dots and 48h window")
    ap.add_argument("--server", default="", help="ERDDAP base URL (defaults to PMEL then AOML)")
    ap.add_argument("--hours", type=int, default=48, help="Track + chart lookback window (hours)")
    ap.add_argument("--size", default="32x64", choices=["32x64","64x128","128x256"], help="Icon size subfolder")
    ap.add_argument("--icons", default="icons", help="Directory containing size folders with pc#.png")
    ap.add_argument("--edge-eps-m", type=float, default=3000.0, help="Inside if within this distance of polygon edge (m)")
    ap.add_argument("--dot-step", type=int, default=20, help="Add a clickable dot every Nth track point")
    ap.add_argument("--dot-lod", type=int, default=128, help="Min LOD pixels for dot visibility")
    ap.add_argument("--dot-region-deg", type=float, default=0.005, help="Region half-size (deg) around each dot")
    ap.add_argument("--out", default=os.path.join("Cstar_Locations", "cstar_locations.kmz"),
                    help="Output KMZ path")
    ap.add_argument("--nhc-cone-url", default="", help="Cone KMZ/KML/GeoJSON URL or local path; leave blank to prompt")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Prompt cone URL if needed
    cone_source = args.nhc_cone_url.strip()
    if not cone_source:
        try:
            cone_source = input("Enter NHC cone URL (KMZ/KML/GeoJSON) or press Enter to auto-detect via NHC Wallets: ").strip()
        except Exception:
            cone_source = ""

    # ERDDAP connect
    servers = [args.server] if args.server else DEFAULT_SERVERS
    server = None
    for s in servers:
        try:
            log(f"🔎 Probing ERDDAP: {s}"); probe_server(s); ok(f"Connected: {s}"); server = s; break
        except Exception as e:
            warn(f"Probe failed: {s} ({e})")
    if not server:
        err("No ERDDAP server reachable from the candidates."); sys.exit(2)

    # Find Oshen datasets
    log("🔎 Searching for 'Oshen' datasets (CSV)…")
    try:
        hits = erddap_search_csv(server, "Oshen")
    except Exception as e:
        err(f"Search failed: {e}"); sys.exit(2)
    dsids: List[str] = []
    for r in hits:
        did = r.get("datasetID") or r.get("Dataset ID") or r.get("DatasetID") or ""
        if did and (did not in dsids):
            dsids.append(did)
    if not dsids:
        err("No 'Oshen' datasets found."); sys.exit(1)
    log(f"Found {len(dsids)} dataset candidates. Verifying structure…")

    platforms: List[Dict[str, Any]] = []
    for did in dsids:
        try:
            info = info_json(server, did) or info_csv(server, did) or info_csvp(server, did)
            if not info: raise RuntimeError("no info")
            ok(f"{did}: metadata gathered")
        except Exception as e:
            warn(f"{did}: info fetch failed ({e})"); continue

        names = list(info.keys()); low = {n.lower() for n in names}
        if not {"time","latitude","longitude"}.issubset(low):
            warn(f"{did}: missing time/lat/lon — skipping"); continue

        id_field  = next((n for n in names if n.lower() in ("trajectoryid","trajectory_id","trajectory","platform","station","id")), None)
        wmo_field = next((n for n in names if n.lower() in ("wmo_id","wmo","wmoid")), None)
        try:
            latest = fetch_latest_position(server, did, id_field, wmo_field)
        except Exception as e:
            warn(f"{did}: latest position fetch failed ({e})"); continue
        if not latest:
            warn(f"{did}: no latest position rows"); continue

        label_guess = str(latest.get(id_field, "")) if id_field else ""
        m = re.search(r"(PC\d+)", label_guess, re.I) or re.search(r"(PC\d+)", did, re.I)
        label = (m.group(1).upper() if m else did)

        platforms.append({
            "dataset_id": did,
            "label": label,
            "id_field": id_field,
            "wmo_field": wmo_field,
            "line_color": LINE_COLORS.get(label, "ff999999"),
            "meta": info,
        })
        ok(f"{did}: ready — label={label}, id_field={id_field}, wmo_field={wmo_field}")

    if not platforms:
        err("No suitable Oshen datasets found."); sys.exit(1)

    # Cone selection (prompt, then wallets)
    cone_polys: List[List[Tuple[float, float]]] = []
    if cone_source:
        log(f"🔎 Loading cone polygons from provided URL/path: {cone_source}")
        cone_polys = load_cone_polys_from_path_or_url(cone_source)
        if cone_polys:
            cone_polys, note = normalize_polygons_longitudes(cone_polys); ok(f"Cone polygons ready ({note}).")
        else:
            ok("No polygons parsed from provided cone; proceeding to wallets/outlook.")
    if not cone_polys and not cone_source:
        alids: Set[str] = set()
        for url in NHC_WALLET_URLS:
            try:
                r = http_get(url, timeout=10)
                ids = sorted(set(re.findall(r'AL\d{2}\d{4}', r.text, re.I)))
                if ids:
                    ok(f"Wallet {url} → AL IDs: {', '.join(ids)}"); alids.update([s.upper() for s in ids])
            except Exception as e:
                warn(f"Wallet fetch failed ({url}): {e}")
        if alids:
            try:
                idx = http_get(NHC_CONE_INDEX, timeout=20).text
                best, best_adv = None, -1
                for al in sorted(alids):
                    pattern = rf'href="({al}_\d+adv_CONE\.kmz)"'
                    for m in re.finditer(pattern, idx, re.I):
                        href = m.group(1)
                        m2 = re.search(r'_(\d+)adv_CONE\.kmz', href, re.I)
                        adv = int(m2.group(1)) if m2 else 0
                        if adv > best_adv:
                            best_adv = adv
                            best = href if href.lower().startswith("http") else (NHC_CONE_INDEX + href)
                if best:
                    log(f"🔎 Loading cone polygons from: {best}")
                    cone_polys = load_cone_polys_from_path_or_url(best)
                    if cone_polys:
                        cone_polys, note = normalize_polygons_longitudes(cone_polys); ok(f"Cone polygons ready ({note}).")
            except Exception as e:
                warn(f"Cone discovery failed: {e}")

    # Outlook polygons
    log("🔎 Fetching NHC outlook polygons…")
    outlook_polys = fetch_nhc_outlook_polygons()
    if outlook_polys:
        outlook_polys, onote = normalize_polygons_longitudes(outlook_polys); ok(f"Outlook polygons ready ({onote}).")

    eps_deg = meters_to_deg(args.edge_eps_m)
    log(f"  Using edge tolerance ≈ {args.edge_eps_m:.1f} m (~{eps_deg:.6f}°)")

    # Cone convex hull (for robust containment)
    cone_hulls: List[List[Tuple[float,float]]] = []
    if cone_polys:
        allpts = [pt for poly in cone_polys for pt in poly]
        hull = convex_hull(allpts)
        if len(hull) >= 3:
            cone_hulls.append(hull); ok("Convex-hull containment enabled for cone.")

    # ---------- Build KML ----------
    doc: List[str] = [kml_header("C-Star Operational — ERDDAP + NHC Cone/Outlook (48h + dots)")]
    labels_used: List[str] = []
    missing_icons: List[str] = []

    # Styles (FIX: alert style IDs now "#{pid}_alert_icon" to match styleUrl)
    for p in platforms:
        pid = p["label"]; labels_used.append(pid)
        kmz_icon = kmz_icon_rel(args.size, pid)
        disk_icon = disk_icon_path(args.icons, args.size, pid)
        doc.append(style_block_icon(pid, kmz_icon, alert_tint=False))
        doc.append(style_block_icon(pid, kmz_icon, alert_tint=True))   # <-- no pid mutation here
        doc.append(style_block_line(pid, p["line_color"]))
        doc.append(style_block_dot(pid))
        doc.append(style_block_alert_ring(pid))
        if not os.path.exists(disk_icon): missing_icons.append(disk_icon)
    if missing_icons:
        warn("Some icons were not found; placeholders will be embedded:")
        for m in missing_icons: warn(f"  missing: {m}")
    else:
        ok("All platform icons found and will be embedded.")

    # Stage dir for charts
    stage_dir = os.path.join(out_dir, "_kmz_stage"); os.makedirs(stage_dir, exist_ok=True)

    impacted_any: Set[str] = set()

    # Variables used in obs tables
    OBS_VARS = [
        "wind_speed_mean",
        "wind_speed_max",
        "wind_from_direction_mean",
        "air_temperature_mean",
        "relative_humidity_mean",
        "sea_surface_temperature_mean",
        "air_pressure_mean",
        "met_averaging_period",
    ]

    for p in platforms:
        did, label = p["dataset_id"], p["label"]
        id_field, wmo_field = p["id_field"], p["wmo_field"]

        doc.append(f'  <Folder><name>{esc(label)} — {esc(did)}</name>\n    <open>1</open>\n')

        # Track for last hours
        try:
            trk = fetch_track(server, did, args.hours)
            if trk:
                ok(f"{label}: track points (last {args.hours}h): {len(trk)}")
                # Track line first (drawn under points)
                doc.append(placemark_track(label, label, p["line_color"], trk, args.hours))
            else:
                warn(f"{label}: no track points in last {args.hours}h")
        except Exception as e:
            warn(f"{label}: track fetch error: {e}")
            trk = []

        # Series for dot obs
        try:
            s_names, s_rows = fetch_vars_series(server, did, OBS_VARS, args.hours)
        except Exception as e:
            warn(f"{label}: obs series fetch failed: {e}")
            s_names, s_rows = [], []

        # Latest + alert + chart
        try:
            latest = fetch_latest_position(server, did, id_field, wmo_field)
            if latest:
                when = str(latest.get("time"))
                lat = float(latest.get("latitude")); lon = float(latest.get("longitude"))
                lon_n = normalize_lon(lon)
                wmo = latest.get(wmo_field) if wmo_field else ""

                # Alert classification
                inside_cone = inside_outlook = False
                if cone_polys:
                    inside_cone = point_in_any_robust(lon_n, lat, cone_polys, eps_deg)
                    if not inside_cone and cone_hulls:
                        if point_in_any_robust(lon_n, lat, cone_hulls, eps_deg):
                            inside_cone = True
                if outlook_polys:
                    inside_outlook = point_in_any_robust(lon_n, lat, outlook_polys, eps_deg)
                show_alert = (inside_cone or inside_outlook)
                if show_alert: impacted_any.add(label)

                # Latest obs row (not necessarily same moment as 'when')
                latest_obs = fetch_latest_obs_vars(server, did, OBS_VARS) or {}
                obs_html_latest = fmt_obs_table_from_row(latest_obs) if latest_obs else "<i>No observation row available.</i>"

                # Composite chart
                chart_name = f"chart_{label}.png"
                chart_written_name = build_composite_chart(
                    server, did, label, args.hours,
                    disk_dir=stage_dir, kmz_rel_name=chart_name,
                    width_px=900, panel_height_px=160, dpi=110
                )
                chart_rel = chart_written_name if chart_written_name else None

                # Track dots (before the main point, so the icon sits on top)
                if trk and s_names and s_rows and args.dot_step > 0:
                    doc.append(f'    <Folder><name>{esc(label)} Track Dots</name>')
                    for (t_iso, lat_i, lon_i) in trk[::args.dot_step]:
                        row = nearest_row_for_time(s_names, s_rows, t_iso, max_tol_sec=900)
                        dot_html = balloon_html_dot(label, t_iso, fmt_obs_table_from_row(row) if row else "<i>No observation row at this time.</i>")
                        doc.append(placemark_track_dot(label, label, t_iso, lat_i, lon_i, dot_html,
                                                       region_deg=args.dot_region_deg, lod_min=args.dot_lod))
                    doc.append('    </Folder>')

                # Latest placemark + optional alert ring
                latest_desc = balloon_html_latest(label, when, lat, lon, wmo, show_alert, obs_html_latest, chart_rel)
                doc.append(placemark_point(label, label, when, latest_desc, lat, lon,
                                           style_suffix=("_alert_icon" if show_alert else "_icon")))
                if show_alert:
                    doc.append(placemark_ring(label, lat, lon, radius_m=12000.0))

            else:
                warn(f"{label}: no latest position")
        except Exception as e:
            warn(f"{label}: latest block error: {e}")

        doc.append("  </Folder>\n")

    # Close KML
    doc.append(kml_footer())
    kml_xml = "".join(doc)

    # ---------- Package KMZ ----------
    out_path = args.out
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # KML
        z.writestr("doc.kml", kml_xml)

        # Icons (platform)
        placeholder = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc````\x00\x00\x00\x05\x00\x01\x0d\n,\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        for pid in labels_used:
            kmz_rel = kmz_icon_rel(args.size, pid)
            disk_rel = disk_icon_path(args.icons, args.size, pid)
            arcname = kmz_rel.replace("\\","/")
            os.makedirs(os.path.dirname(disk_rel), exist_ok=True)
            if os.path.exists(disk_rel):
                with open(disk_rel, "rb") as f:
                    z.writestr(arcname, f.read())
            else:
                z.writestr(arcname, placeholder)

        # Dot icon (small black square)
        dot_png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x05\x00\x00\x00\x05\x08\x06\x00\x00\x00\x8d\x89\x1d\r"
                   b"\x00\x00\x00\x19IDATx\x9ccddd\xfc\xcf\xc0\xc0\xc0\x00\x06\x06\x06\x06\x86\xff\x0f\x03\x03\x03\x00#\x97\x05u"
                   b"\x00\x00\x00\x00IEND\xaeB`\x82")
        z.writestr("dot.png", dot_png)

        # Charts at KMZ root
        stage_dir = os.path.join(out_dir, "_kmz_stage")
        if os.path.isdir(stage_dir):
            for fname in os.listdir(stage_dir):
                if not fname.lower().endswith(".png"): continue
                with open(os.path.join(stage_dir, fname), "rb") as f:
                    z.writestr(fname, f.read())

    ok(f"\nWrote KMZ: {out_path}")

if __name__ == "__main__":
    main()
