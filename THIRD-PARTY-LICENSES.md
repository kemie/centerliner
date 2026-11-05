# Third-party licences

Centerliner's own code is under the [MIT licence](LICENSE). It uses the open-source libraries below, which keep their own licences. This repository contains none of their code: the desktop app installs them with `pip` (see `requirements.txt`), and the browser page loads them from public CDNs (jsDelivr) when you open it.

Checked 2026-11-05 from each package's own licence data. This is a good-faith summary, not legal advice; check the projects themselves if it matters for your use.

## Permissive (BSD, MIT, Apache or similar)
No conditions that matter for using Centerliner.

| Library | Used for | Licence |
|---|---|---|
| numpy | numbers and arrays | BSD-3-Clause (plus small permissive parts) |
| scipy | geometry and fitting maths | BSD-3-Clause |
| scikit-image | medial axis and image clean-up | BSD-3-Clause |
| Pillow | reading rendered images | MIT-CMU |
| Shapely | outline geometry | BSD-3-Clause (see GEOS below) |
| svgpathtools | reading SVG paths | MIT |
| svgelements, svgwrite | SVG reading and writing (desktop) | MIT |
| networkx, imageio, tifffile, lazy_loader, packaging | helpers of the libraries above | BSD or Apache-2.0 |
| resvg_py | Python wrapper around resvg (desktop) | MIT |

## MPL-2.0 (Mozilla Public Licence 2.0)
- **resvg**, the SVG renderer: inside `resvg_py` (desktop) and as `@resvg/resvg-wasm` 2.6.2 (browser page).
- **Pyodide**, Python compiled to run in the browser (browser page).

MPL-2.0 is a file-level licence: it only asks something of anyone who modifies those files and redistributes them. Centerliner modifies and redistributes none of them.

## LGPL-2.1
- **GEOS**, a geometry library that Shapely's ready-made downloads include. Centerliner doesn't ship it either; it comes with Shapely when you install it, or with Pyodide in the browser.

## Not used
AutoTrace (GPL) was only tried once as a separate program in an early comparison. No code from it is in this project.
