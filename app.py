import os
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from paddleocr import PaddleOCR
import gradio as gr

# one‐time model load (≈2–4 s on CPU)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def crop_img(pil_img):
     # 1) Convert PIL→NumPy RGB→BGR for YOLO
    img_np = np.array(pil_img)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2) Run YOLO on the BGR array
    results = model.predict(source=bgr, conf=0.4, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        most_confident = sorted(boxes, key=lambda b: b.conf[0].item(), reverse=True)[0]
        x1, y1, x2, y2 = most_confident.xyxy[0].tolist()
    
        cropped = pil_img.crop((x1, y1, x2, y2))
        cropped_np = np.array(cropped)
        cropped_bgr = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR)
        # save_path = os.path.join(output_folder, f"cropped_{fname}")
        # cropped.save(save_path)
        # print(f"Saved: {save_path}")
    else:
        print(f"No license plate detected in: {fname}")
    
    return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)


def preprocess_plate_improved(gray, debug=False):
    # 0) Upscale 2× with cubic interpolation to give more pixels to work with
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 1) Mild Gaussian blur to remove sensor noise but keep edges
    blurred = cv2.GaussianBlur(upscaled, (5,5), sigmaX=1.0)

    # 2) Contrast stretch (simple linear) instead of CLAHE
    p2, p98 = np.percentile(blurred, (2, 98))
    stretched = cv2.normalize(blurred, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)

    # 3) Unsharp mask via weighted add (stronger than filter2D kernel)
    gauss  = cv2.GaussianBlur(stretched, (0,0), sigmaX=3)
    unsharp = cv2.addWeighted(stretched, 1.5, gauss, -0.5, 0)

    # 4) (Option A) Skip binarization and feed the gray-level unsharp image:
    final = unsharp

    # 4b) (Option B) Or, if you still want binary, use Otsu:
    # _, final = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        imgs = [gray, upscaled, blurred, stretched, unsharp, final]
        titles = ['orig','up×2','blur','stretch','unsharp','final']
        plt.figure(figsize=(12,3))
        for i,(im,t) in enumerate(zip(imgs,titles),1):
            plt.subplot(1,6,i); plt.axis('off'); plt.title(t)
            plt.imshow(im, cmap='gray')
        plt.tight_layout(); plt.show()

    return final

def plate_ocr_paddle(pil_img, debug=False):
    # 1) PIL → NumPy → RGB → gray
    img  = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if debug:
        print("[1] Converted PIL→gray:", gray.shape)

    # 2) Preprocess
    proc = preprocess_plate_improved(gray, debug=debug)
    if debug:
        print("[2] Finished preprocessing")

    # 3) Prepare for PaddleOCR
    rgb_for_ocr = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    if debug:
        print("[3] Converted preprocessed gray→RGB for OCR")

    # 4) Run detection + recognition
    raw = ocr.ocr(rgb_for_ocr, cls=True)
    if debug:
        print("[4] Raw OCR output:", raw)

    # 5) Guard: no output
    if not raw or not raw[0]:
        if debug: print("[5] → No text detected, returning None")
        return None, None
    lines = raw[0]

    # 6) (Optional) Visualize all detections
    if debug:
        vis = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        for bbox, (txt, conf) in lines:
            pts = np.array(bbox, np.int32).reshape((-1,1,2))
            cv2.polylines(vis, [pts], True, (0,0,255), 2)
            x,y = int(bbox[0][0]), int(bbox[0][1]-5)
            cv2.putText(vis, f"{txt}({conf:.2f})", (x,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        plt.figure(figsize=(6,6))
        plt.title("All OCR Detections (red)")
        plt.axis('off')
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.show()

    # 7) Filter & collect candidates
    cands = []
    for idx, item in enumerate(lines):
        try:
            bbox, (txt, conf) = item
        except Exception as e:
            if debug: print(f"[7.{idx}] Bad format, skipping: {item}")
            continue

        # log detection
        if debug: print(f"[7.{idx}] Detected '{txt}' with conf={conf:.2f}")

        # 7a) confidence filter
        if conf < 0.8:
            if debug: print(f"       → Dropped (conf {conf:.2f} < 0.8)")
            continue
        if debug: print("       → Passed confidence filter")

        # 7b) aspect-ratio & area
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        w, h = max(xs)-min(xs), max(ys)-min(ys)
        ar = w/h if h>0 else 0
        if debug:
            print(f"       → bbox size: w={w:.0f}, h={h:.0f}, aspect_ratio={ar:.2f}")
        if h == 0 or ar < 1.0:
            if debug: print(f"       → Dropped (aspect_ratio {ar:.2f} < 1.0)")
            continue
        if debug: print("       → Passed aspect-ratio filter")

        area = w*h
        cands.append((area, txt, conf, bbox))
        if debug: print(f"       → Added candidate: '{txt}', area={area:.0f}")

    # 8) No candidates?
    if not cands:
        if debug: print("[8] No survivors after filtering, returning None")
        return None, None

    # 9) Sort & pick the largest-area candidate
    cands.sort(key=lambda x: x[0], reverse=True)
    area, best_txt, best_conf, best_bbox = cands[0]
    if debug:
        print(f"[9] Selected '{best_txt}' (area={area:.0f}, conf={best_conf:.2f}")

    # Create visualization with the best candidate
    vis2 = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    pts = np.array(best_bbox, np.int32).reshape((-1,1,2))
    cv2.polylines(vis2, [pts], True, (255,0,0), 3)
    x,y = int(best_bbox[0][0]), int(best_bbox[0][1]-5)
    cv2.putText(vis2, f"{best_txt} ({best_conf:.2f})", (x,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    
    # Convert BGR to RGB for display
    vis_rgb = cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB)
    
    return best_txt, vis_rgb

def detect_and_ocr(pil_img):
    # Crop the license plate from the image
    rgb = crop_img(pil_img)
    pil = Image.fromarray(rgb)
    
    # Get the detected text and visualization
    text, vis = plate_ocr_paddle(pil, debug=False)
    
    if text is None:
        print("No license plate text detected")
        return "No text detected", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    print("Detected plate:", text)
    return text, vis  # Return both text and visualization image

# Create the Gradio interface
with gr.Blocks(title="License Plate Reader") as demo:
    gr.Markdown("# YOLO + PaddleOCR License Plate Reader")
    gr.Markdown("Upload a photo of a vehicle to detect and read its license plate.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Vehicle Photo")
            submit_btn = gr.Button("Process")
        
        with gr.Column():
            output_text = gr.Textbox(label="Detected License Plate")
            output_image = gr.Image(label="Cropped License Plate")
    
    submit_btn.click(
        fn=detect_and_ocr,
        inputs=input_image,
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    demo.launch()
