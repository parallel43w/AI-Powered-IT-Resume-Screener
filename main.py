import pandas as pd
import pymupdf
import pytesseract
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from bs4 import BeautifulSoup
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === SECTION 1: Configuration ===

required_sections = {
    "job_title": ["Information Technology", "Cyber Transport", "Supervisor", "Technician"],
    "experience": ["Experience", "Workcenter Supervisor", "Manage", "Troubleshoot"],
    "education": ["Education", "Training", "Degree", "Diploma"],
    "skills": ["Skills", "budget", "networking", "troubleshooting", "routers", "Python", "Cisco", "hardware",
               "communication", "leadership"],
    "additional_info": ["Awards", "Certificates", "Volunteer", "Achievements"]
}

# === SECTION 2: Text Processing ===

lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """ Preprocesses the resume text: lowercasing, removing stopwords, punctuation, and lemmatizing. """
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(words)


def check_section(resume_text, section_keywords):
    return any(keyword.lower() in resume_text.lower() for keyword in section_keywords)


def get_missing_sections(resume_text):
    missing = []
    for section, keywords in required_sections.items():
        if not check_section(resume_text, keywords):
            missing.append(section)
    return missing


def extract_text_from_pdf(pdf_path):
    pdf = pymupdf.open(pdf_path)
    extracted_text = ""
    for page in pdf:
        if page.get_text("text").strip():
            extracted_text += page.get_text()
        else:
            img = page.get_pixmap().pil_image()
            extracted_text += pytesseract.image_to_string(img)
    return extracted_text.strip()


def extract_emails(text):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text):
    pattern = r'https?://\S+|www\.\S+'
    return re.findall(pattern, text)


def extract_skills(text):
    skills_list = ["networking", "troubleshooting", "project management", "Python", "Cisco", "hardware",
                   "communication", "leadership"]
    words = word_tokenize(text.lower())
    return [skill for skill in skills_list if skill in words]


def extract_data_nltk(text):
    return {
        'pages': len(text.split("\f")),
        'emails': extract_emails(text),
        'urls': extract_urls(text),
        'skills': extract_skills(text),
    }


def parse_html(df, column_name="Resume_html", new_column="Resume_Text"):
    df[new_column] = df[column_name].apply(lambda x: BeautifulSoup(str(x), "html.parser").get_text(separator=" "))
    return df


# === SECTION 3: Feature Generation ===

def generate_features(text):
    tfidf_vector = vectorizer.transform([text]).toarray()

    section_features = np.array([
        int(check_section(text, required_sections[section]))
        for section in required_sections
    ]).reshape(1, -1)

    extracted = extract_data_nltk(text)
    extra_features = np.array([
        len(extracted["emails"]),
        len(extracted["skills"]),
        len(extracted["urls"]),
        extracted["pages"]
    ]).reshape(1, -1)

    full_vector = np.hstack([tfidf_vector, section_features, extra_features])
    return full_vector
# === SECTION 4: Training ===

# Load and preprocess data
df = pd.read_csv("Resume.csv", encoding="utf-8")
df = df[df["Category"] == "INFORMATION-TECHNOLOGY"]
df = parse_html(df)

# Label resumes with manual format checker
df["Proper_Format"] = df["Resume_Text"].apply(
    lambda x: int(all(check_section(x, kw) for kw in required_sections.values())))

# Create section flags
for section, keywords in required_sections.items():
    df[section + "_exists"] = df["Resume_Text"].apply(lambda text: int(check_section(text, keywords)))

# Extract extra features
df["num_emails"] = df["Resume_Text"].apply(lambda x: len(extract_emails(x)))
df["num_skills"] = df["Resume_Text"].apply(lambda x: len(extract_skills(x)))
df["num_urls"] = df["Resume_Text"].apply(lambda x: len(extract_urls(x)))
df["num_pages"] = df["Resume_Text"].apply(lambda x: len(x.split("\f")))

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Train the model (use training data)
X_tfidf = vectorizer.fit_transform(df["Resume_Text"]).toarray()

# Combine all features
section_cols = [col for col in df.columns if col.endswith("_exists")]
X_sections = df[section_cols].values
X_extra = df[["num_emails", "num_skills", "num_urls", "num_pages"]].values
X = np.hstack([X_tfidf, X_sections, X_extra])
y = df["Proper_Format"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")


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

    result_str = "\n=== Prediction Report ===\n"
    if is_valid:
        result_str += "✅ Model: Resume is properly formatted.\n"
        if missing:
            result_str += "⚠️  But manually found missing sections:\n"
            for m in missing:
                result_str += f"  - {m.replace('_', ' ').title()}\n"
    else:
        result_str += "❌ Model: Resume is NOT properly formatted.\n"
        if missing:
            result_str += "Missing sections (manual):\n"
            for m in missing:
                result_str += f"  - {m.replace('_', ' ').title()}\n"
        else:
            result_str += "⚠️  Model thinks it's bad, but manual check found all required sections.\n"

    return result_str

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
result = check_resume_format(sample_pdf, vectorizer)  # Pass the vectorizer here
print(f"\nFinal Result: {result}")
