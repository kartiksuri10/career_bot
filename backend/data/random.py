from datetime import datetime
import re
from app.utils import extract_resume
from datetime import datetime
import re

def _get_year_of_completion(education_text):
    if not education_text:
        return None

    ignorePatterns = [r"10th", r"12th", r"ssc", r"hsc", r"secondary school", r"high school"]
    lines = education_text.lower().split("\n")
    filtered_lines = [
        line for line in lines
        if not any(re.search(pat, line) for pat in ignorePatterns)
    ]

    current_year = datetime.now().year
    max_year = current_year + 5

    # Allow either a year or the word 'present'
    range_pattern = re.compile(
        rf"(19[8-9]\d|20[0-9]\d|{max_year})\s*[-–]\s*(19[8-9]\d|20[0-9]\d|{max_year}|present)",
        re.IGNORECASE
    )
    year_pattern = re.compile(rf"\b(19[8-9]\d|20[0-9]\d|{max_year})\b")

    years_found = []

    for line in filtered_lines:
        # First, check for year ranges like '2022 - present'
        for start, end in range_pattern.findall(line):
            if end.lower() == "present":
                return "Present"
            else:
                years_found.append(int(end))

        # Then, check for standalone years
        for match in year_pattern.findall(line):
            year_int = int(match)
            if 1980 <= year_int <= current_year + 1:
                years_found.append(year_int)

    if years_found:
        return max(years_found)
    return None


text = extract_resume.extract_text_from_pdf(r"C:\Users\karti\Downloads\KartikeySinghResume.pdf")
sections = extract_resume.sectionize_resume(text)
print("Year of completion: ",_get_year_of_completion(sections.get('education', '')))