"""Build anacapa_renderer.zip from an explicit list of .py files (no __pycache__)."""
import sys
import zipfile

out = sys.argv[1]
srcs = sys.argv[2:]

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in srcs:
        arcname = "anacapa_renderer/" + f.split("anacapa_renderer/")[-1]
        z.write(f, arcname)

print(f"Packaged {len(srcs)} files → {out}")
