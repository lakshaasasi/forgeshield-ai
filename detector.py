import cv2
import numpy as np
from PIL import Image
import pytesseract
import base64
import os
from pdf2image import convert_from_path
import tempfile

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sasiw\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

LANGUAGES = {
    'eng': 'English',
    'tam': 'Tamil',
    'hin': 'Hindi',
    'tel': 'Telugu',
    'kan': 'Kannada',
    'mal': 'Malayalam'
}

def pdf_to_image(pdf_path):
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        if pages:
            tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            pages[0].save(tmp.name, 'JPEG')
            return tmp.name
    except:
        pass
    return None

def detect_language(image_path):
    try:
        for lang_code, lang_name in LANGUAGES.items():
            text = pytesseract.image_to_string(
                Image.open(image_path), lang=lang_code
            )
            if len(text.strip()) > 30:
                return lang_name, lang_code, text
    except:
        pass
    return 'Unknown', 'eng', ''

def generate_heatmap(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise map
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.absolute(laplacian)
    norm = np.uint8(255 * abs_lap / (abs_lap.max() + 1e-6))

    # Color heatmap
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.45, heatmap, 0.55, 0)

    # Add zones legend on image
    h, w = blended.shape[:2]
    overlay = blended.copy()

    # Add title bar
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 60), -1)
    cv2.putText(overlay, "ForgeShield - Suspicion Heatmap",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)

    # Encode
    _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoded = base64.b64encode(buffer).decode('utf-8')

    # Zone map (simple green/yellow/red)
    zone_map = np.zeros_like(img)
    zone_map[norm < 85] = [0, 180, 0]      # Green = Safe
    zone_map[norm >= 85] = [0, 200, 255]    # Yellow = Moderate
    zone_map[norm >= 170] = [0, 0, 220]     # Red = Suspicious

    zone_blended = cv2.addWeighted(img, 0.5, zone_map, 0.5, 0)
    cv2.putText(zone_blended, "Zone Map: Green=Safe Yellow=Moderate Red=Suspicious",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    _, zbuffer = cv2.imencode('.jpg', zone_blended, [cv2.IMWRITE_JPEG_QUALITY, 90])
    zone_encoded = base64.b64encode(zbuffer).decode('utf-8')

    return encoded, zone_encoded

def get_simple_explanation(reason):
    explanations = {
        "noise": "The document has unusual patterns — like tiny dots or marks that appear when someone edits an image using photo editing software.",
        "edges": "The text borders look uneven — real documents have smooth, consistent edges. This suggests text may have been cut and pasted.",
        "text": "Very little text was found — a real document usually has lots of readable text. This may mean the document was heavily modified.",
        "brightness": "Different parts of the document have different brightness levels — like someone pasted a section from another document.",
        "corrupted": "The file could not be read properly — it may be damaged or in an unsupported format."
    }
    for key, explanation in explanations.items():
        if key in reason.lower():
            return explanation
    return reason

def analyze_document(image_path):
    results = {
        "suspicious": False,
        "reasons": [],
        "simple_reasons": [],
        "confidence": 0,
        "heatmap": None,
        "zone_map": None,
        "language": "Unknown",
        "lang_code": "eng",
        "extracted_text": "",
        "risk_level": "LOW",
        "file_type": "image"
    }

    # Handle PDF
    actual_path = image_path
    if image_path.lower().endswith('.pdf'):
        results["file_type"] = "pdf"
        converted = pdf_to_image(image_path)
        if converted:
            actual_path = converted
        else:
            results["reasons"].append("⚠️ Could not convert PDF")
            results["simple_reasons"].append(get_simple_explanation("corrupted"))
            results["confidence"] = 40
            results["suspicious"] = True
            return results

    img = cv2.imread(actual_path)
    if img is None:
        results["reasons"].append("⚠️ Could not read file")
        results["simple_reasons"].append(get_simple_explanation("corrupted"))
        results["confidence"] = 50
        results["suspicious"] = True
        return results

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check 1: Noise
    noise = cv2.Laplacian(gray, cv2.CV_64F).var()
    if noise > 500:
        results["reasons"].append("⚠️ High noise detected — image may be edited")
        results["simple_reasons"].append(get_simple_explanation("noise"))
        results["confidence"] += 30

    # Check 2: Edges
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size
    if edge_density > 0.1:
        results["reasons"].append("⚠️ Irregular edges — text may be tampered")
        results["simple_reasons"].append(get_simple_explanation("edges"))
        results["confidence"] += 25

    # Check 3: Language + OCR
    lang_name, lang_code, text = detect_language(actual_path)
    results["language"] = lang_name
    results["lang_code"] = lang_code
    results["extracted_text"] = text[:300] if text else ""

    if len(text.strip()) < 20:
        results["reasons"].append("⚠️ Little text detected — document may be altered")
        results["simple_reasons"].append(get_simple_explanation("text"))
        results["confidence"] += 20

    # Check 4: Brightness
    std_bright = np.std(gray)
    if std_bright > 80:
        results["reasons"].append("⚠️ Uneven brightness — possible copy-paste editing")
        results["simple_reasons"].append(get_simple_explanation("brightness"))
        results["confidence"] += 25

    # Risk level
    results["confidence"] = min(results["confidence"], 100)
    if results["confidence"] >= 70:
        results["risk_level"] = "HIGH"
        results["suspicious"] = True
    elif results["confidence"] >= 40:
        results["risk_level"] = "MEDIUM"
        results["suspicious"] = True
    else:
        results["risk_level"] = "LOW"

    # Generate heatmaps
    try:
        results["heatmap"], results["zone_map"] = generate_heatmap(actual_path)
    except:
        pass

    return results