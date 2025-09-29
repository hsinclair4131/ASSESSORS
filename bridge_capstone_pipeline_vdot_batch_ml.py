
from pathlib import Path
import cv2, numpy as np, pandas as pd

ROOT = Path.cwd()
DATA_DIR = ROOT / "BRIDGE CORROSION VDOT"
OUTPUT_DIR = Path("output"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_CSV  = OUTPUT_DIR / "capstone_dataset.csv"
DATASET_XLSX = OUTPUT_DIR / "capstone_dataset.xlsx"

TARGET_SIZE = (1024, 768)
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def imread_any(path: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def ensure_size(img, wh):
    w, h = wh
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def quick_edge_length(gray):
    edges = cv2.Canny(gray, 80, 160)
    kernel = np.ones((3,3), np.uint8)
    thin = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    return float(np.count_nonzero(thin))

def quick_rust_area(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([5,  50,  20], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return float(np.count_nonzero(mask))

def px_to_inches(px, dpi=96.0): return px / dpi

LOW_TAGS = ("LOW",)
MED_TAGS = ("MED", "MEDIUM")
SEV_TAGS = ("SEV", "SEVERE")

def find_one(folder: Path, tags):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in EXTS and any(t in p.stem.upper() for t in tags):
            return p
    raise FileNotFoundError(f"No file with tags {tags} found in {folder}")

def main():
    folders = sorted([p for p in DATA_DIR.glob("IMG *") if p.is_dir()], key=lambda x: x.name)
    rows = []
    for fld in folders:
        p_low = find_one(fld, LOW_TAGS); p_med = find_one(fld, MED_TAGS); p_sev = find_one(fld, SEV_TAGS)
        for label, p in [("Low", p_low), ("Medium", p_med), ("Severe", p_sev)]:
            img = ensure_size(imread_any(p), TARGET_SIZE)
            gray = to_gray(img)
            L = quick_edge_length(gray)
            A = quick_rust_area(img)
            rows.append({
                "Bridge_ID": fld.name,
                "Image_Path": str(p),
                "Risk_Label": label,
                "Crack_Length_in": px_to_inches(L),
                "Rust_Area_in2": A/(96.0*96.0)
            })
    df = pd.DataFrame(rows)
    df.to_csv(DATASET_CSV, index=False)
    df.to_excel(DATASET_XLSX, index=False)
    print("Wrote:", DATASET_CSV.resolve())
    print("Wrote:", DATASET_XLSX.resolve())

if __name__ == "__main__":
    main()
