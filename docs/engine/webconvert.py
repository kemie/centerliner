"""Browser glue: the same job as convert() in app/server.py, but called from the page's worker. Returns a JSON string."""
import json
import re

import centerline_exact as C


def warnings_for(svg_text, ink):
    # same checks as app/server.py (kept in step by hand)
    out = []
    colours = set(re.findall(r'(?:fill|stroke)\s*[:=]\s*["\']?\s*(#[0-9a-fA-F]{3,8}|rgb\([^)]*\)|[a-zA-Z]+)', svg_text))
    colours = {c.lower() for c in colours} - {"none", "currentcolor", "transparent", "inherit", "evenodd", "nonzero", "round", "butt", "square", "miter", "bevel"}
    if len(colours) > 1:
        out.append("This file seems to use several colours. Only one colour is supported; the result uses the dominant one.")
    if "<text" in svg_text:
        out.append("This file contains live text. Convert text to outlines first, or it will be ignored.")
    if re.fullmatch(r"#[0-9a-fA-F]{6}", ink or ""):
        r, g, b = (int(ink[i:i + 2], 16) for i in (1, 3, 5))
        if 0.299 * r + 0.587 * g + 0.114 * b > 200:
            out.append("The artwork's colour is very light, which can confuse the tool. Try it in a dark colour.")
    return out


def convert(svg_text, radius):
    radius = min(max(float(radius), 0.0), 3.0)
    with open("/in.svg", "w", encoding="utf-8") as f:
        f.write(svg_text)
    try:
        res = C.run("/in.svg", corner_radius=radius)
    except ValueError:
        return json.dumps(dict(error="This SVG has no viewBox or width/height, so I can't tell its size."))
    except Exception as e:  # noqa: BLE001 - show any engine failure as a message instead of a crashed page
        return json.dumps(dict(error=f"Could not convert this file ({type(e).__name__}). It may be unsupported."))
    sc = C.reconstruction_scores(res)
    return json.dumps(dict(
        result=C.result_svg(res),
        compare=C.overlay_svg("/in.svg", res, size=900),
        stats=dict(paths=len(res["paths"]), dots=len(res["dots"]), anchors=C.anchors(res),
                   stroke=round(res["w_px"] / res["S"], 2), match=round(sc["iou"] * 100)),
        warnings=warnings_for(svg_text, res.get("ink")) + ([] if res["paths"] or res["dots"] else ["Nothing was found to convert."]),
    ))
