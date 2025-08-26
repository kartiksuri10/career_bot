import fitz #PyMuPDF
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def sectionize_resume(text):
    SECTION_PATTERNS = {
        "summary": r"(summary|objective|profile)",
        "work_experience": r"(work experience|professional experience|employment history|career history|work history)",
        "projects": r"(projects|academic projects|personal projects)",
        "skills": r"(skills|technical skills|key skills|competencies)",
        "education": r"(education|academic background|qualifications|academic details)",
        "certifications": r"(certifications|licenses|accreditations|licences)"
    }
    sections = {}
    text_lower = text.lower()

    # Combine all patterns into a single regex for finding headers
    headers = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in SECTION_PATTERNS.items()}

    # Find positions of section headers
    positions = []
    for section, pattern in headers.items():
        for match in pattern.finditer(text_lower):
            positions.append((match.start(), section))
    
    positions.sort()  # Sort by order of appearance

    # Extract section text
    for i, (start, section) in enumerate(positions):
        end = positions[i+1][0] if i + 1 < len(positions) else len(text)
        sections[section] = text[start:end].strip()
    
    return sections

def preprocess_new(file_path: str) -> str:
    text = extract_text_from_pdf(file_path)
    text = re.sub(r'\s+', ' ', text)
    cleaned_text = text.strip()
    return cleaned_text

def preprocess(file_path: str) -> str:
    global STOP_WORDS
    nlp = spacy.load("en_core_web_sm")
    text = extract_text_from_pdf(file_path)
    sections = sectionize_resume(text)
    text = (
        sections.get('work_experience', '') + " " +
        sections.get('projects', '') + " " +
        sections.get('skills', '') + " " +
        sections.get('certifications', '') + " "
    )
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]
    custom_stop_words = {
        # 1. Resume Section Headers and Structural Words
        "resume", "cv", "summary", "objective", "profile",
        "experience", "work", "professional",
        "education", "academic",
        "skills", "technical", "technologies", "tools",
        "projects", "portfolio",
        "achievements", "accomplishments", "awards", "honors",
        "responsibilities", "duties", "role", "roles",
        "contact", "information", "address", "phone", "email", "linkedin", "github",
        "references", "request", "available", "upon",

        # 2. Generic Action Verbs and Role Descriptors
        "worked", "work",
        "responsible", "responsible for",
        "tasked", "tasked with",
        "duties", "included",
        "involved", "involved in",
        "assisted", "helped", "supported",
        "day-to-day", "daily",
        "ensured", "ensuring",
        "provided", "providing",
        "utilized", "using",

        # 3. Self-Descriptive "Fluff" and Buzzwords
        "motivated", "self-motivated", "driven",
        "hardworking", "dedicated", "passionate",
        "team-player", "collaborative",
        "results-oriented", "results-driven",
        "dynamic", "proactive",
        "excellent", "strong", "proficient", "solid", "good",
        "effective", "efficient",
        "creative", "innovative",
        "detail-oriented",
        "successful", "successfully",

        # 4. General Business and Company Jargon
        "company", "corporation", "corp", "inc", "ltd", "llc", "pvt",
        "business", "organization", "firm",
        "client", "clients", "customer", "customers",
        "team", "department", "group",
        "environment", "setting",
        "industry", "sector",
        "various", "multiple", "numerous",
        "including", "etc", "such as",

        # 5. Time and Measurement Words
        "years", "year",
        "months", "month",
        "new", "current"
    }
    STOP_WORDS |= custom_stop_words
    tokens = [
        # token.lemma_.lower()
        token.text.lower()
        for token in doc
        if token.is_alpha and token.text.lower() not in STOP_WORDS
    ]
    return " ".join(tokens)

