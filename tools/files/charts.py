"""Generate Python (matplotlib) chart images from DataFrames."""

from __future__ import annotations

import base64
import io
from typing import Any

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import pandas as pd

_CJK_FONT_CANDIDATES = (
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
)

_BUNDLED_FONT_PATHS = (
    Path(__file__).resolve().parents[2] / "assets" / "fonts" / "wqy-zenhei.ttc",
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
)


def _configure_matplotlib_fonts() -> None:
    """Pick a font that can render Chinese logistics labels when available."""
    for font_path in _BUNDLED_FONT_PATHS:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
        except (OSError, RuntimeError, ValueError):
            continue

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in _CJK_FONT_CANDIDATES if name in available), "DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen, *_CJK_FONT_CANDIDATES]


_configure_matplotlib_fonts()


def build_charts(frame: pd.DataFrame, *, question: str = "", max_charts: int = 3) -> list[dict[str, Any]]:
    """Inspect a dataframe and render appropriate chart images with matplotlib."""
    if frame is None or frame.empty:
        return []

    normalized = question.lower()
    categorical = _categorical_columns(frame)
    numeric = _numeric_columns(frame)
    datetime_cols = _datetime_columns(frame)
    charts: list[dict[str, Any]] = []

    if datetime_cols and numeric:
        charts.append(
            _line_chart(
                frame,
                x=datetime_cols[0],
                y=numeric[0],
                title=f"{numeric[0]} over time",
            )
        )

    if categorical and numeric:
        aggregated = (
            frame.groupby(categorical[0], dropna=False)[numeric[0]]
            .sum(numeric_only=True)
            .reset_index()
            .sort_values(numeric[0], ascending=False)
            .head(12)
        )
        charts.append(
            _bar_chart(
                aggregated,
                x=categorical[0],
                y=numeric[0],
                title=f"{numeric[0]} by {categorical[0]}",
            )
        )

    status_col = next(
        (
            column
            for column in categorical
            if "status" in column.lower() or "状态" in str(column)
        ),
        None,
    )
    if status_col:
        counts = frame[status_col].value_counts(dropna=False).reset_index()
        counts.columns = [status_col, "count"]
        charts.append(
            _pie_chart(
                counts.head(8),
                label=status_col,
                value="count",
                title=f"{status_col} distribution",
            )
        )

    if len(numeric) >= 2 and ("scatter" in normalized or "correlation" in normalized or not charts):
        charts.append(
            _scatter_chart(
                frame,
                x=numeric[0],
                y=numeric[1],
                title=f"{numeric[0]} vs {numeric[1]}",
            )
        )

    if numeric and ("histogram" in normalized or "distribution" in normalized or len(charts) < 2):
        charts.append(
            _histogram(
                frame,
                column=numeric[0],
                title=f"{numeric[0]} distribution",
            )
        )

    if len(numeric) >= 2 and ("heatmap" in normalized or "matrix" in normalized):
        charts.append(_heatmap(frame, numeric[:6], title="Numeric correlation matrix"))

    ordered = _prioritize([chart for chart in charts if chart], normalized)
    deduped: list[dict[str, Any]] = []
    seen_types: set[str] = set()
    for chart in ordered:
        if chart["type"] in seen_types and chart["type"] != "bar":
            continue
        seen_types.add(chart["type"])
        deduped.append(chart)
        if len(deduped) >= max_charts:
            break
    return deduped


def _prioritize(charts: list[dict[str, Any]], normalized: str) -> list[dict[str, Any]]:
    preferred = None
    if any(word in normalized for word in ("pie", "share", "percent", "status")):
        preferred = "pie"
    elif any(word in normalized for word in ("line", "trend", "over time", "time")):
        preferred = "line"
    elif any(word in normalized for word in ("scatter", "correlation")):
        preferred = "scatter"
    elif any(word in normalized for word in ("histogram", "distribution")):
        preferred = "histogram"
    elif any(word in normalized for word in ("heatmap", "matrix")):
        preferred = "heatmap"
    elif any(word in normalized for word in ("bar", "rank", "compare", "by ")):
        preferred = "bar"

    if not preferred:
        return charts
    return sorted(charts, key=lambda chart: 0 if chart["type"] == preferred else 1)


def _bar_chart(frame: pd.DataFrame, *, x: str, y: str, title: str) -> dict[str, Any] | None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return None
    plot_frame = frame[[x, y]].copy()
    plot_frame[y] = pd.to_numeric(plot_frame[y], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[y])
    if plot_frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = plot_frame[x].astype(str)
    ax.barh(labels[::-1], plot_frame[y][::-1], color="#2F6FED")
    ax.set_xlabel(y)
    ax.set_ylabel(x)
    ax.set_title(title)
    fig.tight_layout()
    return _chart_payload("bar", title, fig, x=x, y=y, data=_records(plot_frame))


def _line_chart(frame: pd.DataFrame, *, x: str, y: str, title: str) -> dict[str, Any] | None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return None
    ordered = frame[[x, y]].copy()
    ordered[x] = pd.to_datetime(ordered[x], errors="coerce")
    ordered[y] = pd.to_numeric(ordered[y], errors="coerce")
    ordered = ordered.dropna().sort_values(x)
    if ordered.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ordered[x], ordered[y], color="#2F6FED", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _chart_payload("line", title, fig, x=x, y=y, data=_records(ordered))


def _pie_chart(frame: pd.DataFrame, *, label: str, value: str, title: str) -> dict[str, Any] | None:
    if frame.empty or label not in frame.columns or value not in frame.columns:
        return None
    plot_frame = frame[[label, value]].copy()
    plot_frame[value] = pd.to_numeric(plot_frame[value], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[value])
    if plot_frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.pie(
        plot_frame[value],
        labels=plot_frame[label].astype(str),
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.Blues(_pie_color_stops(len(plot_frame))),
    )
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    return _chart_payload(
        "pie",
        title,
        fig,
        label=label,
        value=value,
        data=_records(plot_frame),
    )


def _scatter_chart(frame: pd.DataFrame, *, x: str, y: str, title: str) -> dict[str, Any] | None:
    if frame.empty or x not in frame.columns or y not in frame.columns:
        return None
    plot_frame = frame[[x, y]].copy()
    plot_frame[x] = pd.to_numeric(plot_frame[x], errors="coerce")
    plot_frame[y] = pd.to_numeric(plot_frame[y], errors="coerce")
    plot_frame = plot_frame.dropna().head(500)
    if plot_frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(plot_frame[x], plot_frame[y], alpha=0.65, color="#2F6FED", edgecolors="none")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    return _chart_payload("scatter", title, fig, x=x, y=y, data=_records(plot_frame.head(200)))


def _histogram(frame: pd.DataFrame, *, column: str, title: str) -> dict[str, Any] | None:
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None

    bins = min(20, max(5, int(series.nunique())))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    counts, edges, _ = ax.hist(series, bins=bins, color="#2F6FED", edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()

    rows = [
        {
            "bin": f"{edges[index]:.4g} – {edges[index + 1]:.4g}",
            "count": int(counts[index]),
        }
        for index in range(len(counts))
    ]
    return _chart_payload("histogram", title, fig, x="bin", y="count", data=rows)


def _heatmap(frame: pd.DataFrame, columns: list[str], *, title: str) -> dict[str, Any] | None:
    numeric_frame = frame[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric_frame.corr(numeric_only=True)
    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels([str(column) for column in corr.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(column) for column in corr.index])
    ax.set_title(title)
    for row_index in range(len(corr.index)):
        for col_index in range(len(corr.columns)):
            value = corr.values[row_index, col_index]
            if pd.isna(value):
                continue
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _chart_payload(
        "heatmap",
        title,
        fig,
        x="feature",
        y=list(corr.columns),
        data=corr.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
    )


def _chart_payload(
    chart_type: str,
    title: str,
    fig: plt.Figure,
    *,
    data: list[dict[str, Any]] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    image_base64 = _figure_to_base64(fig)
    payload: dict[str, Any] = {
        "type": chart_type,
        "title": title,
        "format": "png",
        "renderer": "matplotlib",
        "image_base64": image_base64,
        "data": data or [],
    }
    payload.update(fields)
    return payload


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    try:
        fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")
    finally:
        plt.close(fig)


def _pie_color_stops(count: int) -> list[float]:
    if count <= 1:
        return [0.55]
    return [0.35 + (0.55 * index / max(count - 1, 1)) for index in range(count)]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Prefer low-cardinality business dimensions; skip unique IDs like order numbers."""
    row_count = max(len(frame), 1)
    columns: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        unique_count = int(series.nunique(dropna=True))
        if unique_count <= 1:
            continue
        if unique_count > min(40, max(12, int(row_count * 0.35))):
            continue
        if _looks_like_id_column(str(column)):
            continue
        columns.append(column)

    preferred_names = {
        "hub",
        "driver",
        "driver_name",
        "customer",
        "customer_name",
        "status",
        "warehouse",
        "城市",
        "州",
        "揽收状态",
        "司机",
        "揽收站点",
        "dock",
    }
    preferred = [
        column
        for column in columns
        if str(column).lower() in preferred_names or str(column) in preferred_names
    ]
    return preferred + [column for column in columns if column not in preferred]


def _looks_like_id_column(column: str) -> bool:
    normalized = column.lower().replace(" ", "")
    id_tokens = (
        "id",
        "单号",
        "运单",
        "任务编码",
        "order",
        "waybill",
        "tracking",
        "code",
        "编码",
    )
    return any(token in normalized for token in id_tokens)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    preferred = (
        "package_count",
        "packages",
        "pickup_count",
        "pickups",
        "completion_rate",
        "delay_rate",
        "failure_rate",
        "delayed_pickups",
        "failed_pickups",
        "总重量",
        "计划揽收包裹数",
        "实际揽收包裹数",
    )
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    # Also promote object columns that are mostly numeric (common in Excel exports).
    for column in frame.columns:
        if column in numeric:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(2, int(len(frame) * 0.6)):
            numeric.append(column)

    ordered = [column for column in preferred if column in numeric]
    ordered.extend(column for column in numeric if column not in ordered)
    return ordered


def _datetime_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            columns.append(column)
            continue
        name = str(column).lower()
        if any(token in name for token in ("date", "time", "day", "timestamp", "时间", "日期")):
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().sum() >= max(2, int(len(frame) * 0.5)):
                columns.append(column)
    return columns


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].astype(str)
    return cleaned.where(pd.notnull(cleaned), None).to_dict(orient="records")
