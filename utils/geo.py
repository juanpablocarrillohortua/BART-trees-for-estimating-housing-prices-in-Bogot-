"""Geographic helpers for the FincaRaiz EDA."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd


class BBox(NamedTuple):
    """Axis-aligned lat/lon box, decimal degrees (WGS84)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


#: The whole Cundinamarca department, rounded outward. Bogota D.C. sits
#: inside it, so Soacha, Chia and the rest of the savanna municipalities
#: are all admitted. A rectangle, not the real departmental boundary: it
#: also covers slices of Boyaca, Tolima and Meta. Deliberately generous
#: -- it is here to catch coordinates that are plainly broken, not to
#: decide which municipality a listing belongs to.
CUNDINAMARCA_BBOX = BBox(
    lat_min=3.65,  # Cabrera, southern tip
    lat_max=5.85,  # northern border with Boyaca
    lon_min=-75.00,  # Magdalena river, western border
    lon_max=-73.00,  # Medina, eastern border with Meta/Casanare
)


def inside_bbox_mask(
    lat: pd.Series,
    lon: pd.Series,
    bbox: BBox = CUNDINAMARCA_BBOX,
) -> pd.Series:
    """Flag the points that fall inside ``bbox``, edges included.

    Rows with a missing latitude or longitude come back ``False``: a
    null is not evidence of being inside the box.
    """
    return lat.between(bbox.lat_min, bbox.lat_max) & lon.between(
        bbox.lon_min, bbox.lon_max
    )


def single_bbox_inside(lat: int, lon: int, bbox: BBox) -> bool:
    """Check a single one coordinate"""

    return (bbox.lat_min < lat < bbox.lat_max) and (
        bbox.lon_min < lon < bbox.lon_max
    )


def outside_bbox_mask(
    lat: pd.Series,
    lon: pd.Series,
    bbox: BBox = CUNDINAMARCA_BBOX,
) -> pd.Series:
    """Flag the points that fall outside ``bbox``.

    Not simply ``~inside_bbox_mask(...)``. ``Series.between`` returns
    ``False`` for ``NaN``, so negating it would report every missing
    coordinate as outside the box. Nulls are excluded here and counted
    separately by :func:`count_outside_bogota`.
    """
    known = lat.notna() & lon.notna()
    return ~inside_bbox_mask(lat, lon, bbox) & known


def count_outside_bogota(
    lat: pd.Series,
    lon: pd.Series,
    bbox: BBox = CUNDINAMARCA_BBOX,
) -> dict[str, Any]:
    """Summarise how many points miss the Bogota region.

    The default box spans the whole Cundinamarca department, so a point
    is only flagged when its coordinates are implausible for the region
    altogether -- not when it merely sits in a neighbouring municipality
    such as Soacha, Chia or Zipaquira, all of which count as inside.

    Returns ``total``, ``inside``, ``outside``, ``missing``,
    ``pct_outside`` (share of ``total``, 2 dp) and the ``bbox`` used.
    ``inside + outside + missing`` always equals ``total``.
    """
    total = len(lat)
    missing = int((lat.isna() | lon.isna()).sum())
    outside = int(outside_bbox_mask(lat, lon, bbox).sum())
    return {
        "total": total,
        "inside": total - outside - missing,
        "outside": outside,
        "missing": missing,
        "pct_outside": round(100 * outside / total, 2) if total else 0.0,
        "bbox": bbox,
    }


# ---------------------------------------------------------------------------
# Localities and municipalities (ported from notebooks/data_cleaning.ipynb)
# ---------------------------------------------------------------------------

#: Bogota's 20 localities under their official names. The boxes overlap
#: along shared borders; :data:`LOCALITIES_BY_AREA` is the order to use when
#: a point needs a single answer.
LOCALITIES: dict[str, BBox] = {
    # 1 — Usaquén (north, east of Autopista Norte, up to Calle 245)
    "Usaquén": BBox(4.685, 4.815, -74.062, -73.985),
    # 2 — Chapinero (Calle 39 to Calle 100, includes eastern hills)
    "Chapinero": BBox(4.600, 4.690, -74.075, -73.985),
    # 3 — Santa Fe (centre-east, includes Monserrate and rural eastern strip)
    "Santa Fe": BBox(4.565, 4.635, -74.095, -74.000),
    # 4 — San Cristóbal (south-east, hillside)
    "San Cristóbal": BBox(4.495, 4.585, -74.115, -74.025),
    # 5 — Usme (south, large rural component)
    "Usme": BBox(4.300, 4.520, -74.175, -74.055),
    # 6 — Tunjuelito (small, south-central)
    "Tunjuelito": BBox(4.545, 4.595, -74.165, -74.115),
    # 7 — Bosa (south-west, along the Bogotá river)
    "Bosa": BBox(4.578, 4.650, -74.225, -74.160),
    # 8 — Kennedy
    "Kennedy": BBox(4.590, 4.665, -74.200, -74.120),
    # 9 — Fontibón (west, includes El Dorado airport)
    "Fontibón": BBox(4.645, 4.720, -74.190, -74.110),
    # 10 — Engativá
    "Engativá": BBox(4.670, 4.750, -74.170, -74.075),
    # 11 — Suba (north-west, large)
    "Suba": BBox(4.700, 4.820, -74.160, -74.025),
    # 12 — Barrios Unidos
    "Barrios Unidos": BBox(4.653, 4.700, -74.095, -74.055),
    # 13 — Teusaquillo
    "Teusaquillo": BBox(4.615, 4.665, -74.105, -74.060),
    # 14 — Los Mártires
    "Los Mártires": BBox(4.590, 4.620, -74.100, -74.065),
    # 15 — Antonio Nariño
    "Antonio Nariño": BBox(4.570, 4.600, -74.105, -74.075),
    # 16 — Puente Aranda
    "Puente Aranda": BBox(4.595, 4.640, -74.130, -74.085),
    # 17 — La Candelaria (historic centre, smallest locality)
    "La Candelaria": BBox(4.588, 4.605, -74.081, -74.065),
    # 18 — Rafael Uribe Uribe
    "Rafael Uribe Uribe": BBox(4.535, 4.585, -74.130, -74.085),
    # 19 — Ciudad Bolívar (south-west, large, steep)
    "Ciudad Bolívar": BBox(4.440, 4.600, -74.220, -74.110),
    # 20 — Sumapaz (rural páramo, by far the largest, extends deep south)
    "Sumapaz": BBox(3.720, 4.320, -74.450, -74.100),
}

#: Smallest box first, so La Candelaria beats Santa Fe in the overlaps.
LOCALITIES_BY_AREA: dict[str, BBox] = dict(
    sorted(
        LOCALITIES.items(),
        key=lambda kv: (
            (kv[1].lat_max - kv[1].lat_min) * (kv[1].lon_max - kv[1].lon_min)
        ),
    )
)

#: The site's spelling -> the key in :data:`LOCALITIES`. The site drops the
#: accents and shortens two names; without this map a lookup by the site's
#: label silently misses half the localities, Usaquén included.
LOCALITY_ALIASES: dict[str, str] = {
    "Usaquen": "Usaquén",
    "Santafe": "Santa Fe",
    "San Cristobal": "San Cristóbal",
    "Fontibon": "Fontibón",
    "Engativa": "Engativá",
    "Los Martires": "Los Mártires",
    "LA Candelaria": "La Candelaria",
    "Rafael Uribe": "Rafael Uribe Uribe",
    "Ciudad Bolivar": "Ciudad Bolívar",
}

#: Ordered smallest -> largest: first match wins, so big boxes can't steal
#: points from the small municipalities they overlap.
MUNICIPALITIES: dict[str, BBox] = {
    "Cajicá": BBox(4.885, 4.955, -74.075, -73.985),
    "Chía": BBox(4.815, 4.895, -74.100, -74.010),
    "Sopó": BBox(4.845, 4.985, -74.000, -73.860),
    "Zipaquirá": BBox(4.940, 5.070, -74.090, -73.940),
    "Soacha": BBox(4.440, 4.640, -74.330, -74.200),
    "Bogotá": BBox(4.460, 4.840, -74.230, -73.980),  # urban, no Sumapaz
}


def locality_price_outliers(
    df: pd.DataFrame,
    multiplier: float = 2.5,
    boxes: dict[str, BBox] = LOCALITIES,
) -> pd.Series:
    """Flag prices beyond ``multiplier`` IQR of the row's own locality.

    The quartiles of ``price_amount`` come from the ``locality`` label, and
    a row is tested only when its coordinates fall inside its own
    locality's box. The boxes overlap, so testing a row against every box
    it falls in would judge it by a neighbour's fences: Engativá's, built
    from few listings, would flag 1,591 rows in Suba and Barrios Unidos,
    none of them in Engativá. Labels must already use the
    :data:`LOCALITIES` keys (see :data:`LOCALITY_ALIASES`).
    """
    lat, lon, price = df["latitude"], df["longitude"], df["price_amount"]
    grouped = df.groupby("locality")["price_amount"]
    q1, q3 = grouped.quantile(0.25), grouped.quantile(0.75)
    low = q1 - multiplier * (q3 - q1)
    high = q3 + multiplier * (q3 - q1)

    weird = pd.Series(False, index=df.index)
    for name, b in boxes.items():
        own = (df["locality"] == name) & inside_bbox_mask(lat, lon, b)
        hi = high.get(name, np.inf)
        lo = low.get(name, -np.inf)
        weird |= own & ((price > hi) | (price < lo))
    return weird


def classify_city(
    lat: pd.Series,
    lon: pd.Series,
    boxes: dict[str, BBox] = MUNICIPALITIES,
    region: BBox = CUNDINAMARCA_BBOX,
) -> pd.Series:
    """Name the municipality each point falls in; first matching box wins.

    Points outside ``region``, or inside it but in no box, come back as
    ``NaN`` -- not ``pd.NA``, which ``SimpleImputer`` does not recognise as
    missing in a string column.
    """
    inside = inside_bbox_mask(lat, lon, region)
    city = pd.Series(np.nan, index=lat.index, dtype="str")
    for name, b in boxes.items():
        hit = inside & inside_bbox_mask(lat, lon, b) & city.isna()
        city = city.mask(hit, name)
    return city


def fill_locality(
    locality: pd.Series,
    lat: pd.Series,
    lon: pd.Series,
    boxes: dict[str, BBox] = LOCALITIES_BY_AREA,
) -> pd.Series:
    """Fill missing localities with the first box the point falls in."""
    filled = locality.copy()
    for name, b in boxes.items():
        hit = filled.isna() & inside_bbox_mask(lat, lon, b)
        filled = filled.mask(hit, name)
    return filled


def snap_locality(
    locality: pd.Series,
    lat: pd.Series,
    lon: pd.Series,
    boxes: dict[str, BBox] = LOCALITIES,
    max_deg: float = 0.02,
) -> pd.Series:
    """Assign the remaining NaNs to the nearest box edge, within ``max_deg``.

    For points that :func:`fill_locality` could not place because they sit
    just outside every box. Anything farther than ``max_deg`` degrees stays
    ``NaN``: those coordinates are wrong, not merely off the edge.
    """
    snapped = locality.copy()
    left = snapped.isna() & lat.notna() & lon.notna()
    for i in snapped.index[left]:
        y, x = lat.at[i], lon.at[i]
        name, d = min(
            (
                (
                    n,
                    np.hypot(
                        max(b.lat_min - y, 0, y - b.lat_max),
                        max(b.lon_min - x, 0, x - b.lon_max),
                    ),
                )
                for n, b in boxes.items()
            ),
            key=lambda t: t[1],
        )
        if d <= max_deg:
            snapped.at[i] = name
    return snapped


def add_landmark_distances(
    df_houses: pd.DataFrame,
    df_landmarks: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Calcula y agrega a `df_houses` la distancia (en km) al landmark positivo

    y negativo más cercano.
    """
    # Separar landmarks por tipo
    pos_landmarks = df_landmarks[df_landmarks["label"] == "positive"][
        [lat_col, lon_col]
    ].to_numpy()
    neg_landmarks = df_landmarks[df_landmarks["label"] == "negative"][
        [lat_col, lon_col]
    ].to_numpy()

    # Coordenadas de las viviendas
    houses_coords = df_houses[[lat_col, lon_col]].to_numpy()

    def min_haversine_distances(coords_a, coords_b):
        """Distancia mínima en km de cada punto de A al conjunto B.

        usando broadcasting de NumPy.
        """
        earth_radius_km = 6371.0

        # Convertir a radianes
        lat1, lon1 = np.radians(coords_a[:, 0]), np.radians(coords_a[:, 1])
        lat2, lon2 = np.radians(coords_b[:, 0]), np.radians(coords_b[:, 1])

        # Matriz 2D: Filas = Viviendas, Columnas = Landmarks
        dlat = lat1[:, np.newaxis] - lat2
        dlon = lon1[:, np.newaxis] - lon2

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1)[:, np.newaxis]
            * np.cos(lat2)
            * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Retorna la distancia mínima (al landmark más cercano) por cada
        # vivienda
        return np.min(earth_radius_km * c, axis=1)

    # Crear una copia del DataFrame para no modificar el original directamente
    df_result = df_houses.copy()

    # Asignar las 2 nuevas columnas
    df_result["dist_nearest_positive_km"] = min_haversine_distances(
        houses_coords, pos_landmarks
    )
    df_result["dist_nearest_negative_km"] = min_haversine_distances(
        houses_coords, neg_landmarks
    )

    return df_result
