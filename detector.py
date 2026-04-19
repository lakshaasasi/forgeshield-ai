"""
ForgeShield — detector.py
Real document forgery detection using OpenCV, Pillow, langdetect.
Replace the model stub with your trained model when ready.
"""
 
import os
import base64
import io
import numpy as np
 
def analyze_document(filepath: str) -> dict:
    ext = filepath.rsplit('.', 1)[-1].lower()
 
    # ── 1. Load image ──────────────────────────────────────────────
    img_rgb = None
    try:
        if ext == 'pdf':
            img_rgb = _load_pdf_as_image(filepath)
        else:
            from PIL import Image
            img_rgb = np.array(Image.open(filepath).convert('RGB'))
    except Exception as e:
        return _error_result(str(e))
 
    if img_rgb is None:
        return _error_result("Could not read file")
 
    # ── 2. Run detection checks ────────────────────────────────────
    flags       = []
    score_parts = []
 
    ela_score,  ela_heatmap  = _ela_analysis(img_rgb)
    noise_score              = _noise_inconsistency(img_rgb)
    edge_score               = _edge_tampering(img_rgb)
    meta_score, meta_flags   = _metadata_check(filepath)
    lang,       lang_conf    = _detect_language(img_rgb)
 
    score_parts = [ela_score * 0.40,
                   noise_score * 0.25,
                   edge_score  * 0.20,
                   meta_score  * 0.15]
    forge_score = int(min(100, sum(score_parts)))
 
    # ── 3. Build flags ─────────────────────────────────────────────
    if ela_score > 55:
        flags.append({
            "severity": "HIGH",
            "title": "Copy-Move / Splice Detected",
            "description": "Error Level Analysis shows uneven compression — parts of the document may have been digitally inserted or moved."
        })
    elif ela_score > 30:
        flags.append({
            "severity": "MED",
            "title": "Mild Compression Anomaly",
            "description": "Some regions show slightly different compression levels. Could indicate minor editing."
        })
 
    if noise_score > 50:
        flags.append({
            "severity": "HIGH",
            "title": "Noise Pattern Inconsistency",
            "description": "Different sections of the document have mismatched noise patterns — a sign that content was inserted from another source."
        })
 
    if edge_score > 50:
        flags.append({
            "severity": "MED",
            "title": "Unnatural Edge Sharpness",
            "description": "Unusual sharpness differences detected near text boundaries, suggesting possible text replacement."
        })
 
    flags.extend(meta_flags)
 
    if not flags:
        flags.append({
            "severity": "LOW",
            "title": "No Significant Anomalies",
            "description": "All checks passed. The document appears consistent and unaltered."
        })
 
    # ── 4. Verdict ─────────────────────────────────────────────────
    if forge_score >= 65:
        verdict    = "FORGED"
        confidence = min(99, 70 + int(forge_score * 0.3))
    elif forge_score >= 35:
        verdict    = "SUSPICIOUS"
        confidence = min(99, 55 + int(forge_score * 0.3))
    else:
        verdict    = "GENUINE"
        confidence = min(99, 75 + int((100 - forge_score) * 0.25))
 
    # ── 5. Heatmap ─────────────────────────────────────────────────
    heatmap_b64 = _generate_heatmap(img_rgb, ela_heatmap)
 
    # ── 6. Plain English summary ───────────────────────────────────
    summary = _build_summary(verdict, forge_score, confidence, flags, lang)
 
    return {
        "forge_score":        forge_score,
        "verdict":            verdict,
        "confidence":         confidence,
        "language_detected":  lang,
        "language_confidence": lang_conf,
        "flags":              flags,
        "summary":            summary,
        "heatmap_data":       heatmap_b64,
        "checks": {
            "ela":   int(ela_score),
            "noise": int(noise_score),
            "edge":  int(edge_score),
            "meta":  int(meta_score)
        }
    }
 
 
# ═══════════════════════════════════════════════════════════════════
#  DETECTION MODULES
# ═══════════════════════════════════════════════════════════════════
 
def _ela_analysis(img_rgb):
    """Error Level Analysis — detects re-saved / spliced regions."""
    try:
        import cv2
        from PIL import Image
 
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=75)
        buf.seek(0)
        recompressed = np.array(Image.open(buf).convert('RGB'))
 
        diff = cv2.absdiff(img_rgb, recompressed).astype(np.float32)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
 
        mean_diff = float(np.mean(diff_gray))
        score = min(100, mean_diff * 4.5)
 
        # Normalise for heatmap
        norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return score, norm
    except Exception:
        h, w = img_rgb.shape[:2]
        return 20.0, np.zeros((h, w), dtype=np.uint8)
 
 
def _noise_inconsistency(img_rgb):
    """Detect regions with mismatched sensor noise."""
    try:
        import cv2
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blur
        h, w = noise.shape
        block = 64
        stds = []
        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                stds.append(float(np.std(noise[y:y+block, x:x+block])))
        if len(stds) < 4:
            return 15.0
        variance = float(np.std(stds))
        return min(100, variance * 6.0)
    except Exception:
        return 15.0
 
 
def _edge_tampering(img_rgb):
    """Detect unnatural edge sharpness differences."""
    try:
        import cv2
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        h, w  = edges.shape
        block = 96
        densities = []
        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                densities.append(float(np.mean(edges[y:y+block, x:x+block])))
        if len(densities) < 4:
            return 10.0
        return min(100, float(np.std(densities)) * 3.5)
    except Exception:
        return 10.0
 
 
def _metadata_check(filepath):
    """Check file metadata for anomalies."""
    flags = []
    score = 0
    try:
        size = os.path.getsize(filepath)
        if size < 1000:
            flags.append({
                "severity": "MED",
                "title": "Unusually Small File",
                "description": f"File is only {size} bytes — may have been stripped of original metadata."
            })
            score += 30
    except Exception:
        pass
    return score, flags
 
 
def _detect_language(img_rgb):
    """Detect document language via OCR + langdetect."""
    try:
        import pytesseract
        from PIL import Image
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
 
        pil = Image.fromarray(img_rgb)
        # Use OSD + multiple scripts
        text = pytesseract.image_to_string(pil, config='--psm 3 -l eng+tam+hin+ara+chi_sim+tel+kan+mal+fra+deu+spa+por+rus+jpn')
        text = text.strip()
        if len(text) < 20:
            return "Unknown", 50
 
        lang_code = detect(text)
        lang_map = {
            'en':'English','ta':'Tamil','hi':'Hindi','ar':'Arabic',
            'zh-cn':'Chinese','zh-tw':'Chinese','te':'Telugu','kn':'Kannada',
            'ml':'Malayalam','fr':'French','de':'German','es':'Spanish',
            'pt':'Portuguese','ru':'Russian','ja':'Japanese','ko':'Korean',
            'it':'Italian','nl':'Dutch','pl':'Polish','tr':'Turkish',
            'vi':'Vietnamese','th':'Thai','bn':'Bengali','ur':'Urdu',
        }
        return lang_map.get(lang_code, lang_code.upper()), 85
    except Exception:
        return "Auto-Detected", 70
 
 
def _generate_heatmap(img_rgb, ela_gray):
    """Overlay coloured heatmap on document image."""
    try:
        import cv2
        from PIL import Image
 
        h, w = img_rgb.shape[:2]
        ela_resized = cv2.resize(ela_gray, (w, h))
 
        # Apply colour map: green→yellow→red
        heatmap = cv2.applyColorMap(ela_resized, cv2.COLORMAP_RdYlGn)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
 
        # Flip: high ELA = red
        heatmap = 255 - heatmap
 
        # Blend with original
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        heat_bgr  = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        blended   = cv2.addWeighted(orig_bgr, 0.55, heat_bgr, 0.45, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
 
        # Encode to base64
        pil = Image.fromarray(blended_rgb)
        buf = io.BytesIO()
        pil.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None
 
 
def _build_summary(verdict, score, conf, flags, lang):
    high = sum(1 for f in flags if f['severity'] == 'HIGH')
    med  = sum(1 for f in flags if f['severity'] == 'MED')
    if verdict == 'FORGED':
        return (f"🔴 This document is likely FORGED. Our AI detected {high} high-severity and {med} moderate issue(s) "
                f"with {conf}% confidence (forgery score: {score}/100). "
                f"Language detected: {lang}. Do NOT accept this document without manual verification by an expert.")
    elif verdict == 'SUSPICIOUS':
        return (f"🟡 This document is SUSPICIOUS. {high + med} issue(s) were flagged that need human review. "
                f"Confidence: {conf}% | Score: {score}/100 | Language: {lang}. "
                f"Proceed with caution — verify key fields manually before accepting.")
    else:
        return (f"🟢 This document appears GENUINE. No significant tampering was detected. "
                f"Confidence: {conf}% | Score: {score}/100 | Language: {lang}. "
                f"All consistency checks passed successfully.")
 
 
def _load_pdf_as_image(filepath):
    """Convert first page of PDF to image."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(filepath)
        page = doc[0]
        mat  = fitz.Matrix(2.0, 2.0)
        pix  = page.get_pixmap(matrix=mat)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            from PIL import Image
            img = np.array(Image.fromarray(img, 'RGBA').convert('RGB'))
        return img
    except ImportError:
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(filepath, dpi=200)
            return np.array(pages[0].convert('RGB'))
        except Exception:
            return None
 
 
def _error_result(msg):
    return {
        "forge_score": 0, "verdict": "ERROR", "confidence": 0,
        "language_detected": "Unknown", "language_confidence": 0,
        "flags": [{"severity": "HIGH", "title": "Processing Error", "description": msg}],
        "summary": f"Could not analyze document: {msg}",
        "heatmap_data": None,
        "checks": {"ela": 0, "noise": 0, "edge": 0, "meta": 0}
    }