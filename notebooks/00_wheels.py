# %% [markdown]
# # 00 — Offline wheels for DICOM decoding
# **Settings:** Accelerator = None (CPU) · Internet = ON · run once.
#
# Submissions run with Internet OFF, so packages we pip-installed during preprocessing (JPEG / JPEG-2000
# decoders for pydicom) must come from files. This notebook downloads them into its output folder `wheels/`;
# the submission notebook attaches this output and installs them with `pip install --no-index`.
#
# v2 (2026-10-11): also downloads the decoder wheels for **other Python versions** (3.11, 3.13). Forked public
# notebooks stay pinned to their author's container image; the 0.954 notebook runs Python 3.13, where our
# cp312 wheels are "not a supported wheel on this platform".

# %%
import os, subprocess, sys

PKGS = ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"]
OTHER_PYTHONS = ["3.11", "3.12", "3.13"]
OUT = "/kaggle/working/wheels"
os.makedirs(OUT, exist_ok=True)
print(sys.version)
r = subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "-d", OUT, *PKGS],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
for ver in OTHER_PYTHONS:          # binary wheels only, no dependencies (numpy is already in every Kaggle image)
    for pkg in PKGS:
        r = subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
                            "--python-version", ver, "--implementation", "cp",
                            "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_17_x86_64",
                            "--platform", "manylinux_2_27_x86_64", "--platform", "manylinux_2_28_x86_64",
                            "-d", OUT, pkg], capture_output=True, text=True)
        print(f"python {ver} {pkg}: rc {r.returncode} {r.stderr.strip()[-200:] if r.returncode else ''}")
files = sorted(os.listdir(OUT))
print(len(files), "files:", files)

# %%
# Prove the wheels install without the internet (fresh target dir, --no-index)
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", OUT,
                    "--target", "/tmp/wheel_test", *PKGS], capture_output=True, text=True)
print("offline install rc =", r.returncode)
print(r.stdout[-1500:], r.stderr[-1500:])
by_py = {v: sorted(f for f in files if f"cp{v.replace('.', '')}-" in f) for v in OTHER_PYTHONS}
print("=== SUMMARY wheels === files:", len(files), "| offline install ok:", r.returncode == 0,
      "| per python:", {v: len(f) for v, f in by_py.items()}, "| cp313:", by_py["3.13"], "=== END ===")
