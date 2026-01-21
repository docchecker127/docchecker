import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import sys
import pytesseract
import os
import re
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
if sys.platform.startswith('linux'):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
else:
    # 🔴 VERIFY THIS PATH
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Holidays for Date Calculator
HOLIDAYS = [
    datetime(2025, 1, 1).date(), datetime(2025, 1, 20).date(),
    datetime(2025, 2, 17).date(), datetime(2025, 5, 26).date(),
    datetime(2025, 6, 19).date(), datetime(2025, 7, 4).date(),
    datetime(2025, 9, 1).date(), datetime(2025, 10, 13).date(),
    datetime(2025, 11, 11).date(), datetime(2025, 11, 27).date(),
    datetime(2025, 12, 25).date(), datetime(2026, 1, 1).date(),
    datetime(2026, 1, 19).date()
]

# ==========================================
# 2. CORE LOGIC (Cached for Speed)
# ==========================================

# A. DRIVER LICENSE CHECKER (Restored)
def check_driver_license(text):
    # Look for expiration dates keywords
    if "EXP" in text.upper() or "EXPIRES" in text.upper():
        # Try to find a date pattern like MM/DD/YYYY
        date_pattern = r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        matches = re.findall(date_pattern, text)
        if matches:
            # Return the last date found (usually expiry)
            return True, f"Found Date: {matches[-1]}"
        return True, "Expiry keyword found (Check date manually)"
    return False, "No Expiry Found"

# B. SIGNATURE ENGINE
def check_signature_engine(img_array):
    h, w = img_array.shape
    y_start, y_end = int(h * 0.15), int(h * 0.95)
    x_start, x_end = int(w * 0.20), int(w * 0.95)
    roi = img_array[y_start:y_end, x_start:x_end]
    
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    remove_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    clean_roi = cv2.subtract(thresh, remove_lines)
    
    contours, _ = cv2.findContours(clean_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    signature_found = False
    
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if h_box > 0: aspect_ratio = float(w_box) / h_box
        else: aspect_ratio = 0
        
        # Density Check
        roi_chunk = roi[y:y+h_box, x:x+w_box]
        total_pixels = w_box * h_box
        if total_pixels > 0:
            dark_pixels = np.count_nonzero(roi_chunk < 100)
            fill_ratio = dark_pixels / total_pixels
        else: fill_ratio = 0
            
        is_big_enough = (w_box > 60 and h_box > 25)
        is_not_thin_bracket = (aspect_ratio > 0.5)
        is_ink_sparse = (fill_ratio < 0.30)
        
        if is_big_enough and is_not_thin_bracket and is_ink_sparse:
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            signature_found = True
        else:
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 0, 255), 1)
            
    return signature_found, debug_img

# C. MAIN PDF PROCESSOR (The "Cache" Wrapper)
# This @st.cache_data line is the MAGIC FIX for the rerun issue.
@st.cache_data(show_spinner=False)
def process_pdf_file(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    results = []
    
    for i in range(len(doc)):
        page = doc[i]
        
        # 1. OCR for Text (Driver License Check)
        text = page.get_text()
        is_dl_page = "DRIVER LICENSE" in text.upper() or "IDENTIFICATION" in text.upper()
        dl_status = None
        if is_dl_page:
            valid_dl, msg = check_driver_license(text)
            dl_status = msg
        
        # 2. Vision for Signature
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        is_signed, debug_img = check_signature_engine(gray)
        
        # Store result for this page
        results.append({
            "page_num": i + 1,
            "is_signed": is_signed,
            "debug_img": debug_img,
            "is_dl": is_dl_page,
            "dl_status": dl_status
        })
    
    return results

# ==========================================
# 3. DATE LOGIC
# ==========================================
def calculate_rescission(sign_date_str):
    try:
        sign_date_str = sign_date_str.replace("-", "/")
        sign_date = datetime.strptime(sign_date_str, "%m/%d/%Y").date()
        business_days = 0
        curr = sign_date
        while business_days < 3:
            curr += timedelta(days=1)
            if curr.weekday() != 6 and curr not in HOLIDAYS:
                business_days += 1
        return curr.strftime("%m/%d/%Y"), None
    except: return None, "Format: MM/DD/YYYY"

# ==========================================
# 4. UI
# ==========================================
st.set_page_config(page_title="DocChecker AI", page_icon="📝", layout="wide")

# Sidebar (Changing this will NO LONGER reload the PDF scan)
with st.sidebar:
    st.header("📅 Date Calculator")
    sign_date_input = st.text_input("Signing Date", placeholder="01/21/2026")
    if sign_date_input:
        rtc, err = calculate_rescission(sign_date_input)
        if err: st.error(err)
        else: st.success(f"Deadline: {rtc}")

st.title("📝 DocChecker AI")
st.write("Upload Loan Packet. AI checks Signatures & Driver Licenses.")

uploaded_file = st.file_uploader("📂 Drop PDF Here", type=["pdf"])

if uploaded_file is not None:
    # We read the bytes once
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("Processing Document..."):
        # This function is CACHED. It only runs once per file.
        page_results = process_pdf_file(file_bytes)
    
    missed_count = 0
    
    for res in page_results:
        p_num = res["page_num"]
        
        # LOGIC:
        # If it's a DL Page, check text format.
        # If it's a Form Page, check Signature box.
        
        if res["is_dl"]:
            status = f"🆔 DL DETECTED: {res['dl_status']}"
            color = "blue"
            expand = True
        elif res["is_signed"]:
            status = "✅ SIGNED"
            color = "green"
            expand = False
        else:
            status = "❌ NOT SIGNED"
            color = "red"
            expand = True
            missed_count += 1
            
        with st.expander(f"Page {p_num}: {status}", expanded=expand):
            st.markdown(f"### Status: :{color}[{status}]")
            st.image(res["debug_img"], channels="BGR")
            
    if missed_count == 0:
        st.success("🎉 No missing signatures found!")
    else:
        st.error(f"🚨 Found {missed_count} missing signatures.")
