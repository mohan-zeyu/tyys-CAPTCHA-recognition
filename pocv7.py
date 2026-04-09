"""
POC: ddddocr detection + multi-rotation classification + Hungarian optimal assignment.

Solves aj-captcha `clickWord` type CAPTCHAs by:
1. Detecting character bounding boxes with ddddocr
2. Classifying each crop across 55 variants (11 rotations × 5 preprocessings)
3. Building a score matrix of per-character hit counts
4. Using the Hungarian algorithm for globally optimal word-to-crop assignment
"""
import base64, json, sys, time
from collections import Counter
from io import BytesIO

import ddddocr
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment


def preprocess_variants(crop_img: Image.Image, size: int = 64) -> list[bytes]:
    """
    Generate multiple image variants from a single crop for robust OCR.

    For each of the 11 rotation angles, produces 5 preprocessed versions:
      - Original color (resized)
      - Grayscale with autocontrast
      - Binary (dark-on-white) at two thresholds
      - Inverted binary (light-on-white) at two thresholds

    Returns a list of PNG-encoded image bytes (11 × 5 = 55 total).
    """
    results: list[bytes] = []
    rotations: list[int] = [0, -30, -20, -10, 10, 20, 30, -45, 45, -60, 60]

    for angle in rotations:
        # Rotate and resize to uniform dimensions
        rotated: Image.Image = crop_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        resized: Image.Image = rotated.resize((size, size))

        # Variant 1: original color
        buf = BytesIO()
        resized.save(buf, format="PNG")
        results.append(buf.getvalue())

        # Variant 2: grayscale with autocontrast
        gray: Image.Image = ImageOps.autocontrast(ImageOps.grayscale(resized), cutoff=10)
        buf = BytesIO()
        gray.save(buf, format="PNG")
        results.append(buf.getvalue())

        # Variants 3-6: binary and inverted at two thresholds
        arr: np.ndarray = np.array(gray)
        for thresh in [110, 140]:
            binary: Image.Image = Image.fromarray(((arr < thresh) * 255).astype(np.uint8))
            buf = BytesIO()
            binary.save(buf, format="PNG")
            results.append(buf.getvalue())

            inv: Image.Image = Image.fromarray(((arr >= thresh) * 255).astype(np.uint8))
            buf = BytesIO()
            inv.save(buf, format="PNG")
            results.append(buf.getvalue())

    return results


# Load CAPTCHA response

with open(sys.argv[1]) as f:
    rep: dict = json.load(f)["data"]["repData"]

word_list: list[str] = rep["wordList"]
img_bytes: bytes = base64.b64decode(rep["originalImageBase64"])
img: Image.Image = Image.open(BytesIO(img_bytes))
print(f"Image: {img.size}, Words to click: {word_list}")

# Step 1: Detect all character bounding boxes

det: ddddocr.DdddOcr = ddddocr.DdddOcr(det=True, show_ad=False)
ocr: ddddocr.DdddOcr = ddddocr.DdddOcr(show_ad=False)

t0: float = time.time()
boxes: list[list[int]] = det.detection(img_bytes)
print(f"Detection: {len(boxes)} boxes in {time.time()-t0:.3f}s")

draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
crops: list[dict] = []

# Step 2: Classify each crop with 55 variants, accumulate hit counts

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    cx: float = (x1 + x2) / 2
    cy: float = (y1 + y2) / 2
    cropped: Image.Image = img.crop((x1, y1, x2, y2))
    variants: list[bytes] = preprocess_variants(cropped)

    # Count how many variants recognized each character
    char_counter: Counter = Counter()
    for v in variants:
        try:
            t: str = ocr.classification(v).strip()
            for ch in t:
                char_counter[ch] += 1
        except Exception:
            pass

    print(f"  crop#{i} ({cx:.0f},{cy:.0f}) -> {char_counter.most_common(5)}")
    crops.append({"cx": cx, "cy": cy, "box": box, "counts": char_counter})
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

# Step 3: Build score matrix [words × crops]

n_words: int = len(word_list)
n_crops: int = len(crops)
score_matrix: np.ndarray = np.zeros((n_words, n_crops))

for wi, word in enumerate(word_list):
    for ci, c in enumerate(crops):
        score_matrix[wi, ci] = c["counts"].get(word, 0)

# Print the matrix for debugging
print(f"\nScore matrix (rows=words, cols=crops):")
header: str = "".ljust(4) + "".join(f"c{i:<6}" for i in range(n_crops))
print(header)
for wi, word in enumerate(word_list):
    row: str = f"'{word}' " + "".join(f"{score_matrix[wi, ci]:<7.0f}" for ci in range(n_crops))
    print(row)

# Step 4: Hungarian algorithm for optimal 1-to-1 assignment
# linear_sum_assignment minimizes cost, so we negate scores to maximize.

cost: np.ndarray = -score_matrix
row_ind: np.ndarray
col_ind: np.ndarray
row_ind, col_ind = linear_sum_assignment(cost)

print(f"\nOptimal assignment:")
click_points: list[dict | None] = [None] * n_words

for wi, ci in zip(row_ind, col_ind):
    c: dict = crops[ci]
    word: str = word_list[wi]
    score: float = score_matrix[wi, ci]
    confidence: str = "" if score > 0 else " ⚠ LOW CONFIDENCE"
    print(f"  '{word}' -> crop#{ci} ({c['cx']:.0f}, {c['cy']:.0f}) hits={score:.0f}{confidence}")
    click_points[wi] = {"x": round(c["cx"]), "y": round(c["cy"])}

# Output
print(f"\nTotal time: {time.time()-t0:.3f}s")
print(f"Result: {json.dumps(click_points)}")
img.save("captcha_annotated.png")
print("Saved captcha_annotated.png")
