import pdfplumber
import re

pdf_path = "sample_statements/idfc-cc-statement.pdf"

def normalize_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

transactions = []
collecting = False
buffer = ""

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text() or ""
        lines = [normalize_space(l) for l in text.splitlines() if l.strip()]

        for line in lines:
            if re.search(r'\bYOUR TRANSACTIONS\b', line, re.IGNORECASE):
                collecting = True
                continue

            if re.search(r'\b(REWARDS|SUMMARY|IMPORTANT INFORMATION|SPECIAL OFFERS|BENEFITS)\b', line, re.IGNORECASE):
                collecting = False
                continue

            if not collecting:
                continue

            # Skip payment instruction lines
            if re.search(r'Pay via|Pay from|Pay through|Click here|Scan QR|Card integrated|Card Number|Enter Credit Card|Enter IFSC|Add IDFC|to open/download|via Bill desk|from other bank|from other banks|PAYMENT MODES', line, re.IGNORECASE):
                continue
            
            # Skip table headers
            if re.search(r'Transaction Date|Transational Details|FX Transactions', line, re.IGNORECASE):
                continue
            
            # merge continuation lines (multi-line descriptions)
            if re.match(r'^\d{2}/\d{2}/\d{4}', line):  # new transaction starts
                if buffer:
                    transactions.append(buffer.strip())
                buffer = line
            else:
                buffer += " " + line

        # flush last buffer per page
        if buffer:
            transactions.append(buffer.strip())
            buffer = ""

print(f"Total buffers: {len(transactions)}")
print("\nAll buffers with date+amount:")
date_amount_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+([0-9,]+\.\d{1,2})')
for i, buf in enumerate(transactions, 1):
    matches = list(date_amount_pattern.finditer(buf))
    if matches:
        print(f"\nBuffer {i} ({len(matches)} matches): {repr(buf[:100])}")
        for m in matches:
            print(f"  -> Date: {m.group(1)}, Amount: {m.group(2)}")

