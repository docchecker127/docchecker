import streamlit as st
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd  # Needed for the clean table
import tempfile
import os
import sys
from datetime import datetime, timedelta

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DocChecker - Stop Missing Signatures",
    page_icon="✍️",
    layout="wide"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
        .main-title {font-size: 2.5rem; font-weight: 700; color: #0E1117; text-align: center; margin-top: -20px;}
        .sub-title {font-size: 1.2rem; color: #555; text-align: center; margin-bottom: 2rem;}
        .pain-point {background-color: #fff3cd; padding: 15px; border-radius: 8px; border: 1px solid #ffeeba; text-align: center; color: #856404; font-weight: 500;}
        .privacy-box {background-color: #f8f9fa; padding: 20px; border-radius: 10px; font-size: 0.9rem; text-align: left; border: 1px solid #dee2e6; margin-top: 30px;}
        .cta-text {text-align: center; font-weight: bold; font-size: 1.1rem; color: #222;}
        .stButton>button {width: 100%; border-radius: 5px; height: 50px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

# Federal Holidays
HOLIDAYS = [
    datetime(2025, 1, 1).date(), datetime(2025, 1, 20).date(),
    datetime(2025, 2, 17).date(), datetime(2025, 5, 26).date(),
    datetime(2025, 6, 19).date(), datetime(2025, 7, 4).date(),
    datetime(2025, 9, 1).date(), datetime(2025, 10, 13).date(),
    datetime(2025, 11, 11).date(), datetime(2025, 11, 27).date(),
    datetime(2025, 12, 25).date(), datetime(2026, 1, 1).date(),
    datetime(2026, 1, 19).date(),
]

def calculate_rescission(sign_date_str):
    try:
        sign_date_str = sign_date_str.replace("-", "/")
        sign_date = datetime.strptime(sign_date_str, "%m/%d/%Y").date()
        business_days_added = 0
        current_date = sign_date
        while business_days_added < 3:
            current_date += timedelta(days=1)
            if current_date.weekday() != 6 and current_date not in HOLIDAYS:
                business_days_added += 1
        return current_date.strftime("%m/%d/%Y"), None
    except:
        return None, "Invalid Date Format. Use MM/DD/YYYY"

def check_signature_final(img_array):
    h, w = img_array.shape
    # Top 15% (skips logos) to Bottom 95%, Left 20% to Right 95%
    y_start, y_end = int(h * 0.15), int(h * 0.95)
    x_start, x_end = int(w * 0.20), int(w * 0.95)
    roi = img_array[y_start:y_end, x_start:x_end]
    
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # Remove Horizontal Lines
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
        
        roi_chunk = roi[y:y+h_box, x:x+w_box]
        total_pixels = w_box * h_box
        
        if total_pixels > 0:
            dark_pixels = np.count_nonzero(roi_chunk < 100)
            fill_ratio = dark_pixels / total_pixels
        else:
            fill_ratio = 0
            
        is_big_enough = (w_box > 60 and h_box > 25)
        is_not_thin_bracket = (aspect_ratio > 0.5)
        is_ink_sparse = (fill_ratio < 0.30)
        
        if is_big_enough and is_not_thin_bracket and is_ink_sparse:
            cv2.rectangle(debug_img, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
            signature_found = True
            
    return signature_found, debug_img

# --- 4. THE WEB APP UI ---

# SIDEBAR
with st.sidebar:
    st.header("📅 Date Calculator")
    sign_date_input = st.text_input("Signing Date", placeholder="01/20/2026")
    if sign_date_input:
        rtc_date, error = calculate_rescission(sign_date_input)
        if error: st.error(error)
        else: st.success(f"**Deadline:** {rtc_date}")
    st.divider()
    st.info("💡 **Tip:** Upload your loan PDF to scan.")

# MAIN CONTENT
st.markdown('<div class="main-title">Doc<span style="color:#FF4B4B">Checker</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">The "Safety Net" for Loan Signing Agents.<br>Scan your Loan Package before you drop it at FedEx.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pain-point">
    🚫 <b>Stop the "Shame Ride" back to the borrower.</b><br>
    One missing initial causes a Funding Condition. Use DocChecker to spot it instantly.
</div>
<br>
""", unsafe_allow_html=True)

st.markdown('<div class="cta-text">👇 Upload Scanned Loan Package (PDF)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file is not None:
    st.divider()
    st.write("🔍 **Scanning Document (Please Wait)...**")
    
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    total_pages = len(doc)
    
    progress_bar = st.progress(0)
    
    # Store results in a list
    results = []
    
    for i in range(total_pages):
        page = doc[i]
        progress_bar.progress((i + 1) / total_pages)
        
        pix = page.get_pixmap(dpi=150)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        is_signed, debug_image = check_signature_final(gray)
        
        # Save data for dashboard
        results.append({
            "Page": i + 1,
            "Status": "✅ Signed" if is_signed else "❌ Missing",
            "Is_Signed": is_signed,
            "Image": debug_image # Keep image in memory to show on demand
        })

    progress_bar.empty()
    
    # Create Dataframe for easy filtering
    df = pd.DataFrame(results)
    missed_count = len(df[df["Is_Signed"] == False])
    
    # --- DASHBOARD UI ---
    
    # 1. Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", total_pages)
    col2.metric("Signed Pages", total_pages - missed_count)
    col3.metric("Missing Signatures", missed_count, delta_color="inverse")
    
    st.markdown("---")
    
    # 2. Tabs for Different Views
    tab1, tab2, tab3 = st.tabs(["🚨 Action Items (Missed)", "📋 Full Audit Table", "🔍 Inspect Any Page"])
    
    # TAB 1: Only show the problems (The "To-Do" List)
    with tab1:
        if missed_count == 0:
            st.success("🎉 No missing signatures found! You are good to go.")
            st.balloons()
        else:
            st.error(f"Found {missed_count} pages that need attention.")
            
            # Show images ONLY for missed pages
            missed_rows = [r for r in results if not r['Is_Signed']]
            for row in missed_rows:
                with st.expander(f"🔴 Page {row['Page']} - Missing Signature", expanded=True):
                    st.image(row['Image'], channels="BGR", use_column_width=True)

    # TAB 2: Clean Table (No Images, just data)
    with tab2:
        st.write("Full list of all pages scanned:")
        # Show simple dataframe (drop the image column for display)
        display_df = df[["Page", "Status"]]
        st.dataframe(display_df, use_container_width=True, height=400)

    # TAB 3: Inspector (View any page image on demand)
    with tab3:
        st.write("Select a page number to verify the scan manually.")
        page_to_view = st.selectbox("Select Page Number:", options=df["Page"].tolist())
        
        # Find the image for that page
        selected_data = next(item for item in results if item["Page"] == page_to_view)
        
        st.write(f"**Status:** {selected_data['Status']}")
        st.image(selected_data['Image'], channels="BGR", caption=f"Page {page_to_view} Analysis")

st.markdown("---")

# --- 5. PRIVACY POLICY ---
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
# --- FOOTER & DISCLAIMER ---
st.markdown("---")
with st.expander("📜 Disclaimer, Privacy & Terms of Use (Read First)", expanded=False):
    st.markdown("""
    **1. No Warranty (As-Is):** This tool is provided for **informational and testing purposes only**. It is built by an indie developer and comes with **NO WARRANTY**. We do not guarantee 100% accuracy. You must manually verify all documents before submission.
    
    **2. Not Legal Advice:** The results from this tool do not constitute legal or professional advice. The developer is not responsible for any rejected loans, funding conditions, or financial losses caused by reliance on this tool.
    
    **3. Privacy & Data Security:** * **Zero Storage:** We do not save, store, or share your files. Files are processed in RAM and deleted immediately after analysis.
    * **User Responsibility:** You act as the Data Controller. Please **REDACT (cover/hide)** sensitive information (SSN, Bank Account Numbers, Names) before uploading if you are concerned about data privacy. By uploading unredacted files, you assume full responsibility for any risks.
    
    **4. Limitation of Liability:** By using this site, you agree to hold the developer harmless from any claims, damages, or losses arising from its use.
    """)
    st.caption("© 2026 Built independently. Not affiliated with any government or banking institution.")
