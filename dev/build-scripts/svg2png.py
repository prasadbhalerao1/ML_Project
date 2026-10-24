import subprocess, sys, re, base64
from pathlib import Path
src, out, scale = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
chrome = sys.argv[4]
for svg in sorted(src.glob("*.svg")):
    t = svg.read_text(encoding="utf-8")
    w = int(re.search(r'width="(\d+)"', t[:300]).group(1))
    h = int(re.search(r'height="(\d+)"', t[:300]).group(1))
    b64 = base64.b64encode(t.encode()).decode()
    html = (f'<html><body style="margin:0;background:#fff">'
            f'<img src="data:image/svg+xml;base64,{b64}" width="{w}" height="{h}">'
            f'</body></html>')
    tmp = out / (svg.stem + ".html")
    tmp.write_text(html, encoding="utf-8")
    png = out / (svg.stem + ".png")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--force-device-scale-factor={scale}",
                    f"--window-size={w},{h}",
                    f"--screenshot={png}", tmp.as_uri()],
                   check=True, capture_output=True)
    tmp.unlink()
    print(png.name, png.stat().st_size // 1024, "KB")
