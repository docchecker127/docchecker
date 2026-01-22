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

# --- 2. CSS STYLING (The Professional Look) ---
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

# --- 5. THE TOOL INTERFACE ---
st.markdown('<div class="cta-text">👇 Upload Scanned Loan Package (PDF)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf", "jpg", "png"])

if uploaded_file is not None:
    st.write("🔄 **Processing Document...**")
# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# A. CLOUD COMPATIBILITY SWITCH
# If running on Linux (Cloud), use default command.
# If Windows (Local), use your specific path.
if sys.platform.startswith('linux'):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
else:
    # 🔴 VERIFY THIS PATH matches your local installation
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# B. FEDERAL HOLIDAYS (For Date Calculator)
# 2025-2026 Holidays (You can expand this list)
HOLIDAYS = [
    datetime(2025, 1, 1).date(),   # New Year
    datetime(2025, 1, 20).date(),  # MLK Day
    datetime(2025, 2, 17).date(),  # Presidents Day
    datetime(2025, 5, 26).date(),  # Memorial Day
    datetime(2025, 6, 19).date(),  # Juneteenth
    datetime(2025, 7, 4).date(),   # Independence Day
    datetime(2025, 9, 1).date(),   # Labor Day
    datetime(2025, 10, 13).date(), # Columbus Day
    datetime(2025, 11, 11).date(), # Veterans Day
    datetime(2025, 11, 27).date(), # Thanksgiving
    datetime(2025, 12, 25).date(), # Christmas
    datetime(2026, 1, 1).date(),   # New Year 2026
    datetime(2026, 1, 19).date(),  # MLK Day 2026
]

# ==========================================
# 2. THE VISION ENGINE (Signature Detection)
# ==========================================
def check_signature_final(img_array):
    h, w = img_array.shape
    
    # --- STEP 1: SCANNING AREA ---
    # Top 15% (skips logos) to Bottom 95%
    # Left 20% (skips punch holes) to Right 95%
    y_start, y_end = int(h * 0.15), int(h * 0.95)
    x_start, x_end = int(w * 0.20), int(w * 0.95)
    
    roi = img_array[y_start:y_end, x_start:x_end]
    
    # --- STEP 2: PRE-PROCESSING ---
    # Adaptive Threshold -> Turns page into Black (Text) & White (Paper)
    # Note: Logic is inverted in OpenCV (White = Ink usually in binary ops)
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # Remove Horizontal Lines (Signature lines, Table borders)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    remove_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    clean_roi = cv2.subtract(thresh, remove_lines)
    
    # --- STEP 3: CONTOUR DETECTION ---
    contours, _ = cv2.findContours(clean_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a color version of the ROI to draw boxes on
    debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    signature_found = False
    
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        
        # Aspect Ratio Check
        if h_box > 0: aspect_ratio = float(w_box) / h_box
        else: aspect_ratio = 0
        
        # --- STEP 4: DENSITY CHECK (The "Anti-Header" Fix) ---
        # We look at the ORIGINAL grayscale pixels inside the box.
        # Header Bars are solid ink (>30% dark pixels).
        # Signatures are scribbles (<30% dark pixels).
        
        roi_chunk = roi[y:y+h_box, x:x+w_box]
        total_pixels = w_box * h_box
        
        if total_pixels > 0:
            # Count pixels darker than 100 (Ink)
            dark_pixels = np.count_nonzero(roi_chunk < 100)
            fill_ratio = dark_pixels / total_pixels
        else:
            fill_ratio = 0
            
        # --- STEP 5: THE FILTER ---
        # 1. Wide enough (> 60px)
        # 2. Tall enough (> 25px)
        # 3. Not a thin vertical bracket (Ratio > 0.5)
        # 4. Not a Solid Block (Fill Ratio < 0.30)
        
        is_big_enough = (w_box > 60 and h_box > 25)
        is_not_thin_bracket = (aspect_ratio > 0.5)
        is_ink_sparse = (fill_ratio < 0.30)
        
        if is_big_enough and is_not_thin_bracket and is_ink_sparse:
            # ✅ GREEN BOX = Valid Signature
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            signature_found = True
        else:
            # 🔴 RED BOX = Ignored (Header bars, tiny text, noise)
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 0, 255), 1)
            
    return signature_found, debug_img

# ==========================================
# 3. DATE LOGIC (Rescission Calculator)
# ==========================================
def calculate_rescission(sign_date_str):
    try:
        # Parse date (handles M/D/Y or M-D-Y)
        sign_date_str = sign_date_str.replace("-", "/")
        sign_date = datetime.strptime(sign_date_str, "%m/%d/%Y").date()
        
        business_days_added = 0
        current_date = sign_date
        
        # Add 3 Business Days (Skipping Sundays & Holidays)
        while business_days_added < 3:
            current_date += timedelta(days=1)
            # Check if Sunday (weekday 6) or Holiday
            if current_date.weekday() != 6 and current_date not in HOLIDAYS:
                business_days_added += 1
                
        return current_date.strftime("%m/%d/%Y"), None
    except Exception as e:
        return None, "Invalid Date Format. Use MM/DD/YYYY"

# ==========================================
# 4. THE WEB APP (Streamlit UI)
# ==========================================
st.set_page_config(page_title="DocChecker AI", page_icon="📝", layout="wide")

# Sidebar: Date Calculator
with st.sidebar:
    st.header("📅 Date Calculator")
    st.write("Check Right-to-Cancel (RTC) Dates")
    sign_date_input = st.text_input("Signing Date", placeholder="01/20/2026")
    
    if sign_date_input:
        rtc_date, error = calculate_rescission(sign_date_input)
        if error:
            st.error(error)
        else:
            st.success(f"**Deadline:** {rtc_date}")
            st.caption("Skips Sundays & Holidays automatically.")
            
    st.divider()
    st.info("💡 **Tip:** Upload your loan PDF on the right to scan for missing signatures.")

# Main Page: Signature Scanner
st.title("📝 DocChecker AI")
st.markdown("""
**California Notaries:** Stop missing signatures.  
Upload your full loan packet below. We scan for missed signatures on Form 1003 and others.
""")

uploaded_file = st.file_uploader("📂 Drop PDF Packet Here", type=["pdf"])

if uploaded_file is not None:
    st.divider()
    st.write("🔍 Scanning document...")
    
    # Open PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    total_pages = len(doc)
    
    # Progress Bar
    progress_bar = st.progress(0)
    
    missed_count = 0
    
    # Create tabs for results
    # We will show "All Pages" or filter to "Missed Only" later? 
    # For now, let's list them cleanly.
    
    for i in range(total_pages):
        page = doc[i]
        
        # Update Progress
        progress_bar.progress((i + 1) / total_pages)
        
        # 1. Convert to Image
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2. Run Vision Engine
        is_signed, debug_image = check_signature_final(gray)
        
        # 3. Display Logic
        status_text = "✅ SIGNED" if is_signed else "❌ NOT SIGNED"
        status_color = "green" if is_signed else "red"
        
        # We use an expander for each page
        # If NOT SIGNED, we open it by default so the user sees it immediately
        with st.expander(f"Page {i+1}: {status_text}", expanded=(not is_signed)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"### Status: :{status_color}[{status_text}]")
                if not is_signed:
                    missed_count += 1
                    st.error("⚠️ Action Required: Signature missing.")
            
            with col2:
                # Convert BGR (OpenCV) to RGB (Streamlit)
                st.image(debug_image, channels="BGR", caption=f"Analysis of Page {i+1}")

    progress_bar.empty()
    
    # Final Summary Report
    st.divider()
    if missed_count == 0:
        st.success("🎉 Perfect! No missing signatures detected in this packet.")
    else:
        st.error(f"🚨 Found {missed_count} pages with missing signatures. Please review above.")
st.markdown("---")

# --- 6. PRIVACY POLICY (Footer) ---
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
