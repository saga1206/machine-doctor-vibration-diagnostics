# Machine Doctor — Non-Contact Vibration Diagnostics

Detects machine vibration/health issues from an ordinary video — no physical
sensors required. Upload or record a short clip of a fan, motor, or other
rotating equipment; the app magnifies vibrations invisible to the naked eye,
filters out camera-shake noise, extracts the true vibration frequency, and
returns a plain-English diagnosis with a downloadable PDF report.

Built as a CPU-only prototype (Intel i7, no GPU) for a non-contact 3D
vibration imaging / predictive maintenance internship application.

## Demo

[![Watch the Demo](https://img.shields.io/badge/📺_Watch_the_Demo-YouTube-blue?style=for-the-badge)](PASTE_YOUR_YOUTUBE_LINK_HERE)

## Screenshots

<table>
<tr>
<td><img src="docs/screenshots/dashboard.png" alt="Dashboard" width="400"/></td>
<td><img src="docs/screenshots/scan-result.png" alt="Scan Result" width="400"/></td>
<td><img src="docs/screenshots/scan-result2.png" alt="Scan Result magnified video" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Fleet Dashboard</b></td>
<td align="center"><b>Scan Result & Diagnosis</b></td>
<td align="center"><b>Motion-Magnified Video</b></td>
</tr>
<tr>
<td><img src="docs/screenshots/machine-history.png" alt="Machine History" width="400"/></td>
<td><img src="docs/screenshots/pdf-report.png" alt="PDF Report" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Trend History</b></td>
<td align="center"><b>PDF Report</b></td>
</tr>
<tr>
<td><img src="docs/screenshots/admin-scans.png" alt="Admin Scans" width="400"/></td>
<td><img src="docs/screenshots/admin-machines.png" alt="Admin Machines" width="400"/></td>
<td><img src="docs/screenshots/admin-machine-fault-patterns.png" alt="Admin Machine-patterns" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Django Admin — Scans</b></td>
<td align="center"><b>Django Admin — Machines</b></td>
<td align="center"><b>Django Admin — Machine Patterns</b></td>
</tr>
</table>

## What it does

1. **Upload or record** a 5-15 second video of a machine (file upload or live webcam)
2. **Ego-motion filtering** — separates your hand-shake from the machine's real vibration using optical flow + RANSAC
3. **Motion magnification** — exaggerates sub-pixel vibration so it's visible in the output video
4. **FFT vibration analysis** — extracts the dominant frequency and amplitude
5. **3D spatial registration** — lightweight two-view depth estimation (Essential matrix + triangulation)
6. **AI diagnosis (RAG)** — retrieves the closest matching fault pattern via embedding similarity, then a real LLM (Gemini) generates a grounded, natural-language diagnosis
7. **History + trend prediction** — compares recent scans per machine, flags a worsening trend before it becomes a failure
8. **PDF report** — downloadable report with chart, diagnosis, and key metrics

## Tech stack

- **Backend:** Django 5, SQLite
- **Vision/Signal processing:** OpenCV (optical flow, RANSAC pose estimation), NumPy, SciPy (FFT, filtering)
- **AI diagnosis:** `sentence-transformers` (embedding retrieval) + Google Gemini API (generation)
- **Video encoding:** `imageio` + `imageio-ffmpeg` (H.264 output, browser-playable)
- **Charts:** Matplotlib
- **PDF reports:** ReportLab
- **Frontend:** Django templates, vanilla JS (webcam recording via `MediaRecorder`)

## Setup

```bash
git clone https://github.com/saga1206/machine-doctor-vibration-diagnostics.git
cd machine-doctor-vibration-diagnostics

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Optional but recommended: real AI-generated diagnosis text.
# Get a free key at https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-key-here"

python manage.py makemigrations
python manage.py migrate
python manage.py seed_fault_patterns
python manage.py createsuperuser

python manage.py runserver
```

Open `http://127.0.0.1:8000/`. Without `GEMINI_API_KEY` set, diagnosis text
falls back automatically to template-based text — the app still fully works,
just without natural-language generation.

## Project structure

```
scanner/
├── models.py                  # Machine, Scan, FaultPattern
├── views.py                   # dashboard, upload, scan_result, machine_detail
├── forms.py
├── analysis/                  # the actual vibration-diagnostics pipeline
│   ├── ego_motion_filter.py   # camera-shake vs. real vibration separation
│   ├── motion_magnify.py      # Eulerian-style motion magnification
│   ├── vibration_fft.py       # frequency/amplitude extraction
│   ├── spatial_3d.py          # two-view Structure-from-Motion
│   ├── diagnosis_rag.py       # embedding-based retrieval
│   ├── generation.py          # LLM-based diagnosis generation (Gemini)
│   ├── trend_prediction.py    # history/trend analysis
│   ├── report_pdf.py          # PDF report generation
│   └── pipeline.py            # orchestrates all of the above
├── templates/scanner/
└── management/commands/seed_fault_patterns.py
fault_knowledge_base.json      # reference fault-pattern knowledge base
```

## Known limitations (by design, not oversights)

- **Amplitude is in relative, uncalibrated units** (pixel-motion), not physical units like mm/s or g. A production system would calibrate this using camera-to-subject distance and a standard like ISO 10816.
- **Frequency ceiling = half the camera's frame rate** (Nyquist limit). A 30fps camera can never reliably detect vibration faster than ~15 Hz — a fundamental limitation of any camera-based approach, not a bug.
- **3D depth needs camera movement.** A perfectly steady tripod shot (ideal for vibration measurement) gives zero parallax for depth estimation. The pipeline automatically tries several frame-gaps and degrades gracefully (marks depth as "unavailable" rather than failing the whole scan) if the footage is too steady.
- **Monocular depth has no absolute scale.** Two cameras (stereo) or a known reference object would be needed to know real-world distances, not just relative depth ordering.
- **Diagnosis is RAG-based, not a trained classifier** — it retrieves the closest matching entry from a small hand-authored knowledge base (6 fault patterns) via embedding similarity, then an LLM writes up the explanation. It's only as good as that knowledge base's coverage.

## License

Personal/educational project — no license specified.