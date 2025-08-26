import fitz  # PyMuPDF
import spacy
from keybert import KeyBERT

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

def extract_text_from_resume(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def extract_skills_and_roles(text):
    doc = nlp(text)
    skills = []
    roles = []

    # Extract keywords
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    keywords = [kw[0] for kw in keywords]

    # Extract roles from entities or keywords
    for ent in doc.ents:
        if ent.label_ in ["ORG", "WORK_OF_ART"]:
            roles.append(ent.text)

    # Use keyword extraction for skills
    for kw in keywords:
        if any(term in kw.lower() for term in ["developer", "engineer", "scientist", "analyst"]):
            roles.append(kw)
        else:
            skills.append(kw)

    return list(set(roles)), list(set(skills))

def generate_queries_from_resume(file_path):
    text = extract_text_from_resume(file_path)
    roles, skills = extract_skills_and_roles(text)

    queries = []
    for role in roles:
        queries.append(f"{role} jobs in India")

    # Add some skill-based queries
    for skill in skills:
        queries.append(f"{skill} jobs in India")

    return list(set(queries))  # remove duplicates
if __name__ == "__main__":
    print(generate_queries_from_resume(r"C:\Users\karti\Downloads\rajneesh_resume.pdf"))