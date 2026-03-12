"""Self-contained HTML visualizations for feature activations.
Inspired by CircuitsVis colored_tokens."""

import html
import uuid


def colored_tokens_html(
    tokens: list[str],
    values: list[float],
    title: str | None = None,
    positive_color: str = "rgba(30, 100, 200, {a})",
    negative_color: str = "rgba(200, 50, 30, {a})",
    max_value: float | None = None,
) -> str:
    """Render tokens with background color proportional to activation value."""
    if len(tokens) != len(values):
        raise ValueError(f"tokens ({len(tokens)}) and values ({len(values)}) must match")

    div_id = f"tb-{uuid.uuid4().hex[:8]}"
    abs_max = max_value or max((abs(v) for v in values), default=1.0) or 1.0

    spans = []
    for token, val in zip(tokens, values):
        alpha = min(abs(val) / abs_max, 1.0) * 0.8
        color = positive_color.format(a=alpha) if val >= 0 else negative_color.format(a=alpha)
        escaped = html.escape(token).replace(" ", "&nbsp;")
        spans.append(
            f'<span style="background:{color};padding:2px 1px;border-radius:3px;" '
            f'title="{val:.2f}">{escaped}</span>'
        )

    title_html = (
        f"<h4 style='margin:4px 0;font-family:monospace'>{html.escape(title)}</h4>" if title else ""
    )

    return (
        f'<div id="{div_id}" style="font-family:monospace;font-size:14px;'
        f'line-height:1.8;padding:8px">\n{title_html}{"".join(spans)}\n</div>'
    )
