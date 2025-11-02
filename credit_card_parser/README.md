# Credit Card Statement Parser

A Streamlit app to parse credit card PDF statements and extract key details.

## Features

- **Multiple Input Options**: Radio button interface to choose between uploading a PDF file or selecting from sample files
- **Password-Protected PDF Support**: 
  - Automatically detects password-protected PDFs when uploaded/selected
  - Prompts for password with a clear input field
  - Validates password and shows clear error messages if incorrect
  - All extraction functions support password-protected PDFs
- **Automatic Processing**: 
  - Non-protected PDFs are processed automatically without user interaction
  - Protected PDFs process automatically once the correct password is entered
  - No need to click a "Process" button
- **Transaction Extraction**: Extracts and displays all transactions with date, description, and amount
- **CSV Export**: Download extracted transactions as CSV file
- **Multi-Bank Support**: Works with statements from multiple banks
- **Smart Error Handling**: Clear error messages for wrong passwords and other processing errors

## Supported Issuers

- HDFC Bank
- ICICI Bank
- Axis Bank
- IDFC FIRST Bank
- Syndicate Bank

## Extracted Fields

- Cardholder Name
- Card Number (last 4 digits)
- Statement Period / Cycle
- Payment Due Date
- Total Amount Due
- (Optional) Transactions table: Date, Description, Amount

## Quickstart

1. Install dependencies:

```bash
pip install -r credit_card_parser/requirements.txt
```

2. Run the app:

```bash
streamlit run credit_card_parser/app.py
```

3. Choose an input method using the radio buttons:

   - **Upload PDF**: Upload your credit card statement PDF file using the file uploader
   - **Choose Sample**: Select from available sample files in `sample_statements/` directory

4. **Password-Protected PDFs**:

   - The app **automatically detects** if your PDF is password-protected when you upload/select it
   - If protected, you'll see a password input field appear
   - Enter the password to unlock the PDF
   - The app will **automatically process** the PDF once the correct password is entered (no button click needed)
   - If the password is incorrect, you'll see a clear **"❌ Wrong password entered. Please check your password and try again."** message

5. View the extracted information and transactions, and download as CSV if needed.

## Password-Protected PDFs

The app includes comprehensive support for password-protected PDFs:

### Automatic Detection
- The app **automatically checks** if a PDF is password-protected when you upload or select it
- Detection happens before processing begins
- No manual indication needed - the app handles it automatically

### Processing Flow
- **Non-protected PDFs**: 
  - Processed automatically immediately after selection
  - No password prompt or user interaction required
  
- **Protected PDFs**:
  - App automatically detects the protection and shows an info message
  - Password input field appears automatically
  - Enter the password to unlock the PDF
  - Processing begins **automatically** once the password is entered (no button click needed)
  
### Error Handling
- **Wrong Password**: Shows clear error message: "❌ Wrong password entered. Please check your password and try again."
- **Missing Password**: Shows warning if password is required but not entered
- **Other Errors**: Displays appropriate error messages for other processing issues

### Supported Methods
- All extraction functions (`extract_text_from_pdf`, `extract_transactions`, issuer-specific extractors) support password-protected PDFs
- Password is passed securely through all extraction layers

## Notes

- Uses `pdfplumber` for text and table extraction.
- OCR (pytesseract + pdf2image) is optional and not enabled by default.
- Regex-based parsing varies by issuer; refine patterns in `parser.py` as needed.
- Password protection detection works with standard PDF encryption methods supported by `pdfplumber`.
- The app handles `PdfminerException` errors that may occur with encrypted PDFs.
- Password input is masked for security (password type input field).

## UI Features

- **Radio Button Selection**: Clear interface to choose between upload and sample file options
- **Automatic Processing**: No need to click "Process" button - extraction happens automatically
- **Password Input**: Secure password input field (masked) when PDF is protected
- **Real-time Feedback**: 
  - Spinner shows during processing
  - Success message when extraction completes
  - Clear error messages for any issues
- **Transaction Display**: Clean dataframe view of all extracted transactions
- **CSV Download**: One-click download of transactions as CSV file

## Project Structure

```
credit_card_parser/
├── app.py
├── parser.py
├── requirements.txt
└── README.md
```
