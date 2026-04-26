#!/usr/bin/env python3
"""Build a self-contained seal_eda_standalone.html with the JSON data embedded.

Reads:
  - seal_eda.html
  - seal_viz_data.json

Writes:
  - seal_eda_standalone.html (sendable as a single file)

The HTML's loadData() prefers an embedded <script id="embedded-data"> block
over fetch(), so the standalone and dev versions share the same source.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML_SRC  = HERE / "seal_eda.html"
DATA_SRC  = HERE / "seal_viz_data.json"
HTML_OUT  = HERE / "seal_eda_standalone.html"

html = HTML_SRC.read_text(encoding="utf-8")
data = DATA_SRC.read_text(encoding="utf-8")

# Protect against any literal "</script>" inside the JSON (unlikely but safe).
data_safe = data.replace("</", "<\\/")

embedded_block = (
    '<script id="embedded-data" type="application/json">\n'
    f"{data_safe}\n"
    "</script>\n"
)

marker = "<body>"
if marker not in html:
    raise SystemExit("Couldn't find <body> tag in seal_eda.html")

html_standalone = html.replace(marker, marker + "\n" + embedded_block, 1)
HTML_OUT.write_text(html_standalone, encoding="utf-8")

size_mb = HTML_OUT.stat().st_size / 1e6
n_keys  = data.count('":')  # rough count, just for the log line
print(f"Wrote {HTML_OUT.name} ({size_mb:.2f} MB)")
print(f"Open with: open {HTML_OUT}")
