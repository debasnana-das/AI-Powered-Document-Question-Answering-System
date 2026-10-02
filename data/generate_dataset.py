from pathlib import Path
import textwrap
import fitz
OUT=Path(__file__).resolve().parent/"sample_documents"/"Employee_Policy.pdf"
OUT.parent.mkdir(parents=True,exist_ok=True)
content=["Employee Remote Work and Leave Policy","Employees may work remotely up to three days per week, subject to manager approval and team coverage requirements.","Annual leave entitlement is twenty days per calendar year for full-time employees.","Remote work requests should be approved by the employee's direct manager before the remote work day.","Employees are expected to attend required onsite meetings and comply with applicable public holiday schedules."]
doc=fitz.open(); page=doc.new_page(); y=72
for index,paragraph in enumerate(content):
    font_size=16 if index==0 else 11
    for line in textwrap.wrap(paragraph,width=85):
        page.insert_text((72,y),line,fontsize=font_size); y+=24 if index==0 else 18
    y+=10
doc.save(OUT); doc.close(); print(f"Created {OUT}")
