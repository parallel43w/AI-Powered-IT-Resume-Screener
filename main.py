import pandas as pd
import pymupdf
import pytesseract
import fitz
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === SECTION 1: Configuration ===

required_sections = {
    "job_title": ["Information Technology", "Technician", "IT Specialist", "Systems Admin"],
    "experience": ["Experience", "Workcenter Supervisor", "Manage", "Responsibilities"],
    "education": ["Education", "Training", "Degree", "Diploma", "University"],
    "skills": ["Skills", "technologies", "tools", "programming", "languages", "Cisco", "Python"],
    "additional_info": ["Awards", "Certifications", "Volunteer", "Achievements", "Accomplishments"]
}

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stopwords.words('english')]
    return ' '.join(words)

def check_section(resume_text, keywords):
    return any(k.lower() in resume_text.lower() for k in keywords)

def get_missing_sections(text):
    return [s for s, k in required_sections.items() if not check_section(text, k)]

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        if page.get_text("text").strip():
            text += page.get_text()
        else:
            img = page.get_pixmap().pil_image()
            text += pytesseract.image_to_string(img)
    return text.strip()

def extract_emails(text):
    return re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

def extract_urls(text):
    return re.findall(r'https?://\S+|www\.\S+', text)

def extract_skills(text):
    skill_keywords = ["networking", "troubleshooting", "project", "Python", "Cisco", "leadership", "firewall"]
    words = word_tokenize(text.lower())
    return [skill for skill in skill_keywords if skill in words]

def extract_data(text):
    return {
        'emails': extract_emails(text),
        'urls': extract_urls(text),
        'skills': extract_skills(text),
        'pages': len(text.split("\f"))
    }



def parse_html(df, column_name="Resume_html", new_column="Resume_Text"):
    df[new_column] = df[column_name].apply(lambda x: BeautifulSoup(str(x), "html.parser").get_text(separator=" "))
    return df


# === SECTION 3: Feature Generation ===

def generate_features(text):
    tfidf_raw = vectorizer.transform([text]).toarray()
    # Pad if needed
    if tfidf_raw.shape[1] < 5000:
        tfidf_raw = np.pad(tfidf_raw, ((0, 0), (0, 5000 - tfidf_raw.shape[1])), 'constant')
    elif tfidf_raw.shape[1] > 5000:
        tfidf_raw = tfidf_raw[:, :5000]

    section_feats = np.array([int(check_section(text, kw)) for kw in required_sections.values()]).reshape(1, -1)
    extracted = extract_data(text)
    extra_feats = np.array([
        len(extracted["emails"]),
        len(extracted["skills"]),
        len(extracted["urls"]),
        extracted["pages"]
    ]).reshape(1, -1)

    return np.hstack([tfidf_raw, section_feats, extra_feats])

# === SECTION 4: Training ===

# Load and preprocess data
df = pd.read_csv("Resume.csv", encoding="utf-8")
df = df[df["Category"] == "INFORMATION-TECHNOLOGY"]
df["Resume_Text"] = df["Resume_html"].apply(lambda x: BeautifulSoup(str(x), "html.parser").get_text(" "))

# Label as properly formatted if 4 or more of 5 sections found
df["Proper_Format"] = df["Resume_Text"].apply(lambda x: int(sum(check_section(x, kw) for kw in required_sections.values()) >= 4))

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(df["Resume_Text"]).toarray()

# Extra features
section_flags = np.array([[int(check_section(text, kw)) for kw in required_sections.values()] for text in df["Resume_Text"]])
extra_feats = np.array([
    df["Resume_Text"].apply(lambda x: len(extract_emails(x))),
    df["Resume_Text"].apply(lambda x: len(extract_skills(x))),
    df["Resume_Text"].apply(lambda x: len(extract_urls(x))),
    df["Resume_Text"].apply(lambda x: len(x.split("\f")))
]).T

X = np.hstack([X_tfidf, section_flags, extra_feats])
y = df["Proper_Format"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))


# === SECTION 5: Prediction Function ===
def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select Resume PDF",
        filetypes=[("PDF files", "*.pdf")]
    )
    if file_path:
        result = check_resume_format(file_path)
        display_result(result)
def display_result(result_text):
    output_box.config(state='normal')
    output_box.delete('1.0', tk.END)
    output_box.insert(tk.END, result_text)
    output_box.config(state='disabled')
def check_resume_format(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    missing = get_missing_sections(text)
    features = generate_features(text)
    is_valid = model.predict(features)[0]

    report = "\n=== Resume Format Check ===\n"
    if is_valid:
        report += "✅ Model: Resume is properly formatted.\n"
        if missing:
            report += "⚠️  But manual check found missing sections:\n"
            for m in missing:
                report += f"  - {m.replace('_', ' ').title()}\n"
    else:
        report += "❌ Model: Resume is NOT properly formatted.\n"
        if not missing:
            report += "⚠️  Manual check: All required sections seem to be present.\n"
        else:
            report += "Missing sections (manual):\n"
            for m in missing:
                report += f"  - {m.replace('_', ' ').title()}\n"
    return report

# === GUI Layout ===

root = tk.Tk()
root.title("Resume Format Checker")
root.geometry("600x400")

title_label = tk.Label(root, text="📄 Resume Format Checker", font=("Arial", 16))
title_label.pack(pady=10)

browse_btn = tk.Button(root, text="Select Resume PDF", command=browse_file, font=("Arial", 12))
browse_btn.pack(pady=5)

output_box = ScrolledText(root, height=15, width=70, wrap=tk.WORD, state='disabled', font=("Courier", 10))
output_box.pack(padx=10, pady=10)

root.mainloop()
# Test the model
sample_pdf = "10089434.pdf"
result = check_resume_format(sample_pdf) # Pass the vectorizer here
print(f"\nFinal Result: {result}")
