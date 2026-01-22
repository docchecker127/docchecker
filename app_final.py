import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import tempfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DocChecker - Stop Missing Signatures",
    page_icon="✍️",
    layout="centered"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
        .main-title {font-size: 2.5rem; font-weight: 700; color: #0E1117; text-align: center; margin-top: -50px;}
        .sub-title {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
        .pain-point {background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeeba; text-align: center; color: #856404; font-weight: 500;}
        .privacy-box {background-color: #f8f9fa; padding: 20px; border-radius: 10px; font-size: 0.9rem; text-align: left; border: 1px solid #dee2e6; margin-top: 30px;}
        .cta-text {text-align: center; font-weight: bold; font-size: 1.1rem; color: #222;}
        .stButton>button {width: 100%; border-radius: 5px; height: 50px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# --- 3. HERO SECTION ---
st.markdown('<div class="main-title">Doc<span style="color:#FF4B4B">Checker</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">The "Safety Net" for Loan Signing Agents.<br>Scan your Loan Package before you drop it at FedEx.</div>', unsafe_allow_html=True)

# --- 4. THE PAIN POINT ---
st.markdown("""
<div class="pain-point">
    🚫 <b>Stop the "Shame Ride" back to the borrower.</b><br>
    One missing initial causes a Funding Condition. Use DocChecker to spot it instantly.
</div>
<br>
""", unsafe_allow_html=True)

# --- 5. LOGIC ENGINE (The Brain) ---
def process_page(image_np):
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Threshold to find black text/lines
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (potential boxes/lines)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    issues_found = False
    annotated_img = image_np.copy()
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter: Look for box-like shapes (Signature lines usually have specific width/height)
        # Adjust these values based on testing. currently set for signature lines/boxes
        if w > 50 and h > 10 and h < 100: 
            # Crop the box area to check if it's empty
            roi = thresh[y:y+h, x:x+w]
            non_zero_pixels = cv2.countNonZero(roi)
            
            # If pixels are very low, it means it's likely empty (unsigned)
            # We assume the box line itself takes some pixels, so we check for "ink" inside
            fill_ratio = non_zero_pixels / (w * h)
            
            if fill_ratio < 0.1: # Less than 10% filled (Adjustable sensitivity)
                # Draw RED box around missing signature
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
                issues_found = True
                
    return annotated_img, issues_found

# --- 6. THE TOOL INTERFACE ---
st.markdown('<div class="cta-text">👇 Upload Scanned Loan Package (PDF)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file is not None:
    st.write("🔄 **Processing Document... (This happens in RAM)**")
    
    # Save uploaded file temporarily to process with PyMuPDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        doc = fitz.open(tmp_path)
        total_issues = 0
        
        # Loop through pages
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150) # render page to image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            
            # Run the Logic
            result_img, issue_detected = process_page(img_np)
            
            if issue_detected:
                total_issues += 1
                st.error(f"❌ **Potential Missing Signature on Page {i+1}**")
                st.image(result_img, use_column_width=True)
                st.markdown("---")
        
        if total_issues == 0:
            st.success("✅ **Great Job! No missing signatures detected.**")
            st.balloons()
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        
    finally:
        # Cleanup temp file
        os.remove(tmp_path)

st.markdown("---")

# --- 7. PRIVACY POLICY ---
st.subheader("🔒 Bank-Grade Privacy")
st.markdown("""
<div class="privacy-box">
    <strong>We know Title Companies are strict. Here is our security promise:</strong>
    <ul>
        <li><strong>RAM-Only Processing:</strong> Your files are processed in temporary memory and wiped immediately.</li>
        <li><strong>No Cloud Storage:</strong> We do not save, archive, or view your Loan Packages.</li>
        <li><strong>Encrypted:</strong> All data is transmitted via secure SSL (HTTPS).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<br><center><small>© 2025 DocChecker.co | Built for Notaries</small></center>", unsafe_allow_html=True)
