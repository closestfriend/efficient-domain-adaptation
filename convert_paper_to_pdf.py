#!/usr/bin/env python3
"""
Convert PAPER_DRAFT.md to PDF with embedded figures
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

# Read the markdown file
md_path = Path("docs/PAPER_DRAFT.md")
with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Fix image paths to be absolute paths for weasyprint
# Convert: ![Figure 1](figures/figure1...png)
# To: ![Figure 1](docs/figures/figure1...png)
md_content = re.sub(
    r'!\[(.*?)\]\(figures/',
    r'![\1](docs/figures/',
    md_content
)

# Convert markdown to HTML
md = markdown.Markdown(extensions=['extra', 'tables', 'toc'])
html_content = md.convert(md_content)

# Create a styled HTML document
html_doc = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: letter;
            margin: 1in;
        }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #000;
        }}
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            margin-top: 24pt;
            margin-bottom: 12pt;
            text-align: center;
        }}
        h2 {{
            font-size: 14pt;
            font-weight: bold;
            margin-top: 18pt;
            margin-bottom: 10pt;
        }}
        h3 {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 14pt;
            margin-bottom: 8pt;
        }}
        p {{
            margin-bottom: 8pt;
            text-align: justify;
        }}
        img {{
            max-width: 100%;
            display: block;
            margin: 12pt auto;
        }}
        table {{
            border-collapse: collapse;
            margin: 12pt auto;
            width: 90%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 6pt;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        code {{
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            background-color: #f5f5f5;
            padding: 2pt 4pt;
        }}
        pre {{
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 10pt;
            border: 1px solid #ddd;
            overflow-x: auto;
        }}
        strong {{
            font-weight: bold;
        }}
        em {{
            font-style: italic;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 24pt 0;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Write HTML to PDF
output_path = "docs/PAPER.pdf"
HTML(string=html_doc, base_url=str(Path.cwd())).write_pdf(output_path)

print(f"✓ PDF generated successfully: {output_path}")
print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
