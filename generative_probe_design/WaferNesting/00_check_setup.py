"""Stage 00 -- check the install before running anything. Run this first.

Resolves every path the selected config depends on and reports what is missing and which
line to edit. Also checks the Python packages and looks for KLayout.

    python3 WaferNesting/00_check_setup.py
    WAFERNEST_CONFIG=config_electrode_bundle python3 WaferNesting/00_check_setup.py

Exits non-zero if anything required is missing.
"""
import importlib, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CFG_NAME = os.environ.get("WAFERNEST_CONFIG", "config")

KLAYOUT_CANDIDATES = [
    "/Applications/klayout.app/Contents/MacOS/klayout",
    "/Applications/KLayout/klayout.app/Contents/MacOS/klayout",
]

problems = []


def ok(label, detail=""):
    print("  OK      %-26s %s" % (label, detail))


def bad(label, detail, fix):
    print("  MISSING %-26s %s" % (label, detail))
    problems.append((label, fix))


print("config          : %s" % CFG_NAME)
try:
    C = importlib.import_module(CFG_NAME)
except Exception as e:
    print("\nFAILED to import config %r: %s" % (CFG_NAME, e))
    sys.exit(1)

print("this folder     : %s" % C.HERE)
print("common folder   : %s" % os.path.dirname(C.HERE))
print("run tag         : %s" % C.RUN_TAG)
print("outputs go to   : %s" % C.OUT_DIR)

# --- python packages ----------------------------------------------------------
print("\nPython packages (stages 01-02):")
for pkg in ("ezdxf", "shapely", "numpy", "matplotlib"):
    try:
        m = importlib.import_module(pkg)
        ok(pkg, getattr(m, "__version__", ""))
    except ImportError:
        bad(pkg, "not importable", "pip install %s" % pkg)

# --- klayout ------------------------------------------------------------------
print("\nKLayout (stages 03-04):")
kl = next((p for p in KLAYOUT_CANDIDATES if os.path.exists(p)), None) or shutil.which("klayout")
if kl:
    ok("klayout", kl)
else:
    bad("klayout", "not found", "install KLayout, or edit KLAYOUT_CANDIDATES in this file")

# --- shared inputs ------------------------------------------------------------
print("\nInputs:")
if os.path.isfile(C.WAFER_TEMPLATE):
    ok("wafer template", C.WAFER_TEMPLATE)
else:
    bad("wafer template", C.WAFER_TEMPLATE,
        "set WAFER_TEMPLATE in %s.py to the wafer GDS" % CFG_NAME)

if os.path.isfile(C.MARKS_WKT):
    ok("alignment marks", C.MARKS_WKT)
else:
    bad("alignment marks", C.MARKS_WKT,
        "set MARKS_WKT in %s.py -- marks_union.wkt ships inside WaferNesting/" % CFG_NAME)

if C.SEED_MODE == "seeded":
    if C.SEED_POSES and os.path.isfile(str(C.SEED_POSES)):
        ok("seed poses", C.SEED_POSES)
    else:
        bad("seed poses", str(C.SEED_POSES),
            "SEED_MODE is 'seeded' -- point SEED_POSES at a poses JSON, or set "
            "SEED_MODE = 'free' in %s.py" % CFG_NAME)

# --- designs ------------------------------------------------------------------
print("\nDesigns (DESIGN_DIR = %s):" % C.DESIGN_DIR)
if not os.path.isdir(C.DESIGN_DIR):
    bad("DESIGN_DIR", "%s is not a folder" % C.DESIGN_DIR,
        "set DESIGN_DIR in %s.py to the folder holding the design DXFs" % CFG_NAME)
for d in C.DESIGNS:
    p = C.design_dxf(d)
    if os.path.isfile(p):
        ok(d["key"], "x%-3d %s" % (d["count"], os.path.basename(p)))
    else:
        bad(d["key"], p,
            "check the 'dxf' entry for %r in %s.py, and DESIGN_DIR above"
            % (d["key"], CFG_NAME))

if os.path.isdir(C.DESIGN_DIR):
    dxfs = sorted(f for f in os.listdir(C.DESIGN_DIR) if f.lower().endswith(".dxf"))
    print("\n  DXFs available in DESIGN_DIR: %s"
          % (", ".join(dxfs) if dxfs else "(none)"))

# --- verdict ------------------------------------------------------------------
total = sum(d["count"] for d in C.DESIGNS)
print("\nWould nest %d piece(s): %s"
      % (total, ", ".join("%d x %s" % (C.count_of(k), k) for k in C.kinds())))

if problems:
    print("\n%d problem(s):" % len(problems))
    for label, fix in problems:
        print("  - %-20s -> %s" % (label, fix))
    sys.exit(1)
print("\nSETUP OK -- run 01_extract_footprints.py next")
