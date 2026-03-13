import math

f_ctrl = 100.0
f_plan = 30.0
T_ctrl = 1.0 / f_ctrl
T_plan = 1.0 / f_plan

# logarithmic frequency samples: 0.1~20Hz
n = 600
f_min, f_max = 0.1, 20.0
freqs = [f_min * ((f_max / f_min) ** (i / (n - 1))) for i in range(n)]

mag_db = []
phase_deg = []
for f in freqs:
    w = 2 * math.pi * f
    x = w * T_plan / 2
    if abs(x) < 1e-12:
        mag = 1.0
    else:
        mag = abs(math.sin(x) / x)
    phase = -(w * T_plan / 2 + w * T_ctrl)
    mag_db.append(20 * math.log10(max(mag, 1e-12)))
    phase_deg.append(math.degrees(phase))

# -3dB bandwidth
f_bw = None
for f, m in zip(freqs, mag_db):
    if m <= -3.0:
        f_bw = f
        break

for fs in [1, 3, 5, 8, 10, 12, 15]:
    ws = 2 * math.pi * fs
    lag_deg = math.degrees(-(ws * T_plan / 2 + ws * T_ctrl))
    print(f"f={fs:>4.1f} Hz, total phase lag~{lag_deg:>7.2f} deg")
print(f"Estimated -3 dB bandwidth (ZOH only): {f_bw:.2f} Hz")

# simple SVG plotting utility
W, H = 1000, 700
margin = 70
plot_w = W - 2 * margin
plot_h_each = (H - 3 * margin) / 2

log_min = math.log10(f_min)
log_max = math.log10(f_max)

mag_min, mag_max = -20, 1
phase_min, phase_max = -240, 0

def x_map(f):
    return margin + (math.log10(f) - log_min) / (log_max - log_min) * plot_w

def y_map(v, vmin, vmax, y0, ph):
    return y0 + ph * (1 - (v - vmin) / (vmax - vmin))

# Build polylines
mag_points = " ".join(
    f"{x_map(f):.2f},{y_map(m, mag_min, mag_max, margin, plot_h_each):.2f}"
    for f, m in zip(freqs, mag_db)
)
phase_y0 = 2 * margin + plot_h_each
phase_points = " ".join(
    f"{x_map(f):.2f},{y_map(p, phase_min, phase_max, phase_y0, plot_h_each):.2f}"
    for f, p in zip(freqs, phase_deg)
)

nyquist = f_plan / 2
x_ny = x_map(nyquist)

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">')
svg.append('<style>text{font-family:Arial,sans-serif;font-size:14px;} .grid{stroke:#ddd;stroke-width:1;} .axis{stroke:#000;stroke-width:1.5;} .curve1{stroke:#1565c0;stroke-width:2;fill:none;} .curve2{stroke:#8e24aa;stroke-width:2;fill:none;} .ref{stroke:#c62828;stroke-dasharray:6 4;} .ny{stroke:#000;stroke-dasharray:3 4;} </style>')
svg.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18">Planner 30 Hz ZOH + 1-step Control Delay @ 100 Hz</text>')

# Axes boxes
svg.append(f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h_each}" fill="none" class="axis"/>')
svg.append(f'<rect x="{margin}" y="{phase_y0}" width="{plot_w}" height="{plot_h_each}" fill="none" class="axis"/>')

# Reference lines
y_m3 = y_map(-3, mag_min, mag_max, margin, plot_h_each)
svg.append(f'<line x1="{margin}" y1="{y_m3:.2f}" x2="{margin+plot_w}" y2="{y_m3:.2f}" class="ref"/>')
svg.append(f'<line x1="{x_ny:.2f}" y1="{margin}" x2="{x_ny:.2f}" y2="{phase_y0+plot_h_each}" class="ny"/>')

if f_bw is not None:
    x_bw = x_map(f_bw)
    svg.append(f'<line x1="{x_bw:.2f}" y1="{margin}" x2="{x_bw:.2f}" y2="{margin+plot_h_each}" stroke="#2e7d32" stroke-dasharray="6 4"/>')
    svg.append(f'<text x="{x_bw+6:.2f}" y="{margin+18}" fill="#2e7d32">-3 dB ≈ {f_bw:.2f} Hz</text>')

# Curves
svg.append(f'<polyline points="{mag_points}" class="curve1"/>')
svg.append(f'<polyline points="{phase_points}" class="curve2"/>')

# Labels
svg.append(f'<text x="15" y="{margin+plot_h_each/2:.2f}" transform="rotate(-90 15,{margin+plot_h_each/2:.2f})">Magnitude (dB)</text>')
svg.append(f'<text x="15" y="{phase_y0+plot_h_each/2:.2f}" transform="rotate(-90 15,{phase_y0+plot_h_each/2:.2f})">Phase (deg)</text>')
svg.append(f'<text x="{W/2}" y="{H-15}" text-anchor="middle">Frequency (Hz, log scale)</text>')
svg.append(f'<text x="{x_ny+6:.2f}" y="{phase_y0+20}" font-size="12">Planner Nyquist 15 Hz</text>')

svg.append('</svg>')

out = 'analysis/bode_tracking_100_30.svg'
with open(out, 'w', encoding='utf-8') as f:
    f.write("\n".join(svg))
print(f'Saved plot: {out}')
