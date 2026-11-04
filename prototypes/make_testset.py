"""Write synthetic test shapes to testset/: NAME.svg (filled outline) and NAME.truth.json (true centerlines)."""
import json
from common import ROOT, build_cases, case_geometry, outline_svg, truth_lines

out = ROOT / "testset"
out.mkdir(exist_ok=True)
for name, spec in build_cases().items():
    g = case_geometry(spec)
    (out / f"{name}.svg").write_text(outline_svg(g))
    (out / f"{name}.truth.json").write_text(json.dumps({"note": spec["note"], "lines": [t.round(2).tolist() for t in truth_lines(spec)]}))
    print(name, "nodes in outline:", sum(len(r.coords) for r in ([g.exterior, *g.interiors] if g.geom_type == "Polygon" else [])))
