"""Re-scrape every source, rebuild the unified dataset and re-run the model."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["scrape_cricbuzz.py", "--refresh"],
    ["scrape_cricsheet.py", "--refresh"],
    ["scrape_espn.py", "--refresh"],
    ["scrape_squads.py"],
    ["scrape_weather.py", "--refresh"],
    ["build_dataset.py"],
    ["model.py"],
]

for step in STEPS:
    print(f"\n=== {' '.join(step)}")
    subprocess.run([sys.executable, os.path.join(HERE, step[0]), *step[1:]], check=True)
