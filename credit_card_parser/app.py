import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path

from parser import (
    extract_text_from_pdf,
    detect_issuer,
    extract_data,
    extract_data_schema,
    extract_transactions,
    extract_hdfc_key_fields,
    extract_icici_key_fields,
    extract_idfc_key_fields,
    extract_syndicate_key_fields,
)

# Page configuration
st.set_page_config(
    page_title="Credit Card Parser",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Parser")
st.markdown("Upload your credit card statement PDF to extract information and transactions. You can also upload pdf that are password protected.")

# Radio button to choose between upload and sample
file_source_option = st.radio(
    "Choose an option:",
    ["Upload PDF", "Choose Sample"],
    horizontal=True,
    label_visibility="visible"
)

st.markdown("---")

uploaded_file = None
selected_sample = None
sample_options = {}

# Show file uploader or sample selector based on radio selection
if file_source_option == "Upload PDF":
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Select a credit card statement PDF file"
    )
else:
    st.markdown("### Sample Files")
    sample_dir = (Path(__file__).resolve().parent.parent / "sample_statements")
    sample_files = []
    if sample_dir.exists():
        sample_files = sorted(sample_dir.glob("*.pdf"))

    if sample_files:
        sample_options = {f.name: str(f) for f in sample_files}
        selected_sample = st.selectbox(
            "Choose a sample file",
            ["Select a sample..."] + list(sample_options.keys())
        )
    else:
        selected_sample = None
        st.info("No sample PDFs available.")

# Process file
if uploaded_file is not None or (selected_sample and selected_sample != "Select a sample..."):
    # Determine which file to use
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
            file_source = "uploaded"
    else:
        temp_path = sample_options[selected_sample]
        file_source = "sample"
    
    # Check if PDF is password-protected
    pdf_password = None
    pdf_requires_password = False
    try:
        import pdfplumber
        # Try to open without password
        with pdfplumber.open(temp_path) as pdf:
            # Try to access first page to ensure it's actually readable
            try:
                _ = pdf.pages[0].extract_text()
            except Exception:
                pass
    except Exception as e:
        # Check if error is related to password/encryption
        error_str = str(e).lower()
        error_type = type(e).__name__.lower()
        
        # Check underlying exception if it's a wrapped exception
        underlying_error = ""
        if hasattr(e, 'args') and e.args:
            underlying_error = str(e.args[0]).lower() if e.args else ""
        
        all_error_text = f"{error_str} {underlying_error}".lower() if underlying_error else error_str
        
        # PdfminerException often indicates password-protected PDF
        # Only treat as password-protected if error explicitly mentions encryption/password or is PdfminerException
        if ("pdfminer" in error_type or  # PdfminerException
            "encrypted" in all_error_text or "decrypt" in all_error_text or "encrypt" in all_error_text or
            "password" in all_error_text or "bad password" in all_error_text or
            "incorrect password" in all_error_text or "invalid password" in all_error_text):
            pdf_requires_password = True
    
    # Function to extract and display data
    def process_pdf(pwd=None):
        with st.spinner("Extracting data from statement..."):
            try:
                # Extract data
                text = extract_text_from_pdf(temp_path, password=pwd)
                issuer = detect_issuer(text)
                data = extract_data(text, issuer)
                schema = extract_data_schema(text, issuer)
                
                # Get issuer-specific fields
                if issuer.upper() == "HDFC":
                    fields = extract_hdfc_key_fields(temp_path, password=pwd)
                    display_map = {
                        "Cardholder Name": fields.get("Cardholder Name"),
                        "Card Number (last 4)": fields.get("Card Number (last 4)"),
                        "Statement Date": fields.get("Statement Date"),
                        "Payment Due Date": fields.get("Payment Due Date"),
                        "Total Amount Due": fields.get("Total Amount Due"),
                    }
                    schema_data = {k: (v if v else "Not found") for k, v in display_map.items()}
                elif issuer.upper() == "ICICI":
                    fields = extract_icici_key_fields(temp_path, password=pwd)
                    display_map = {
                        "Cardholder Name": fields.get("Cardholder Name"),
                        "Card Number (last 4)": fields.get("Card Number (last 4)"),
                        "Statement Date": fields.get("Statement Date"),
                        "Payment Due Date": fields.get("Payment Due Date"),
                        "Total Amount Due": fields.get("Total Amount Due"),
                    }
                    schema_data = {k: (v if v else "Not found") for k, v in display_map.items()}
                elif issuer.upper() == "IDFC":
                    fields = extract_idfc_key_fields(temp_path, password=pwd)
                    display_map = {
                        "Cardholder Name": fields.get("Cardholder Name"),
                        "Card Number (last 4)": fields.get("Card Number (last 4)"),
                        "Statement Period": fields.get("Statement Period"),
                        "Payment Due Date": fields.get("Payment Due Date"),
                        "Total Amount Due": fields.get("Total Amount Due"),
                    }
                    schema_data = {k: (v if v else "Not found") for k, v in display_map.items()}
                elif issuer.upper() == "SYNDICATE":
                    fields = extract_syndicate_key_fields(temp_path, password=pwd)
                    display_map = {
                        "Cardholder Name": fields.get("Cardholder Name"),
                        "Card Number (last 4)": fields.get("Card Number (last 4)"),
                        "Statement Date": fields.get("Statement Date"),
                        "Payment Due Date": fields.get("Payment Due Date"),
                        "Total Amount Due": fields.get("Total Amount Due"),
                    }
                    schema_data = {k: (v if v else "Not found") for k, v in display_map.items()}
                else:
                    schema_data = schema.to_dict()
                
                transactions = extract_transactions(temp_path, password=pwd)
                
                # Display results
                st.success("✅ Extraction complete!")
                st.markdown("---")
                
                # Detected Bank
                st.subheader("🏦 Detected Bank")
                st.write(issuer.upper())
                st.markdown("---")
                
                # Extracted Information
                st.subheader("📋 Extracted Information")
                for field, value in schema_data.items():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**{field}:**")
                    with col2:
                        st.write(value)
                st.markdown("---")
                
                # Transactions
                st.subheader("💰 Transactions")
                if transactions is not None and not transactions.empty:
                    st.dataframe(transactions, use_container_width=True, height=400)
                    
                    # Download button
                    csv = transactions.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Transactions as CSV",
                        csv,
                        "transactions.csv",
                        "text/csv",
                        key="download-csv"
                    )
                else:
                    st.warning("No transactions found in the statement.")
                    
            except ValueError as e:
                # ValueError from extract_text_from_pdf indicates password error
                error_msg = str(e).lower()
                if "wrong password" in error_msg or "incorrect" in error_msg:
                    st.error("❌ Wrong password entered. Please check your password and try again.")
                elif "password" in error_msg or "encrypt" in error_msg:
                    st.error(f"❌ {str(e)}")
                else:
                    st.error(f"Error: {str(e)}")
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__.lower()
                
                # Check underlying exception if it's a wrapped exception
                underlying_error = ""
                if hasattr(e, 'args') and e.args:
                    underlying_error = str(e.args[0]).lower() if e.args else ""
                
                all_error_text = f"{error_str} {underlying_error}".lower() if underlying_error else error_str
                
                # Check if it's a password-related error (wrong password)
                # If PDF requires password and we got PdfminerException, it's likely wrong password
                if ("wrong password" in all_error_text or 
                    "incorrect password" in all_error_text or 
                    "invalid password" in all_error_text or
                    "bad password" in all_error_text or
                    ("password" in all_error_text and "encrypted" in all_error_text) or
                    ("pdfminer" in error_type and pdf_requires_password)):  # If PDF requires password and got PdfminerException
                    st.error("❌ Wrong password entered. Please check your password and try again.")
                elif "encrypted" in all_error_text or "decrypt" in all_error_text or "password" in all_error_text:
                    st.error("❌ PDF encryption error. Please check your password and try again.")
                else:
                    st.error(f"Error processing file: {str(e)}")
                    st.exception(e)
    
    # If PDF is not password-protected, automatically process
    if not pdf_requires_password:
        process_pdf()
    else:
        # If PDF is password-protected, ask for password first
        st.info("🔒 This PDF is password-protected. Please enter the password below to continue.")
        pdf_password = st.text_input(
            "PDF Password:",
            type="password",
            help="Enter the password to unlock the PDF file",
            label_visibility="visible",
            key="pdf_password_input"
        )
        
        # Process when password is entered
        if pdf_password:
            process_pdf(pdf_password)
        else:
            st.warning("⚠️ Please enter the PDF password to proceed.")
