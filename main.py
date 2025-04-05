import pandas as pd
import pymupdf
import pytesseract
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to check if a section exists in the resume
def check_section(resume_text, section_keywords):
    return any(keyword.lower() in resume_text.lower() for keyword in section_keywords)

# Define expected sections in a resume
required_sections = {
    "job_title": ["Information Technology", "Cyber Transport", "Supervisor", "Technician"],
    "experience": ["Experience", "Workcenter Supervisor", "Manage", "Troubleshoot"],
    "education": ["Education", "Training", "Degree", "Diploma"],
    "skills": ["Skills", "budget", "networking", "troubleshooting", "routers"],
    "additional_info": ["Awards", "Certificates", "Volunteer", "Achievements"]
}

# Function to check if the resume has the necessary sections
def check_format(resume_text):
    return all(check_section(resume_text, keywords) for _, keywords in required_sections.items())

# Function to extract text from a PDF resume
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

# Function to extract emails using regex
def extract_emails(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails

# Function to extract URLs using regex
def extract_urls(text):
    url_pattern = r'https?://\S+|www\.\S+'
    urls = re.findall(url_pattern, text)
    return urls

# Function to extract skills using keyword matching
def extract_skills(text):
    skills_list = ["networking", "troubleshooting", "project management", "Python", "Cisco", "hardware"]
    words = word_tokenize(text.lower())
    extracted_skills = [skill for skill in skills_list if skill in words]
    return extracted_skills

# Function to extract important details from resume text
def extract_data_nltk(text):
    return {
        'pages': len(text.split("\f")),  # Count pages based on form feeds
        'emails': extract_emails(text),
        'urls': extract_urls(text),
        'skills': extract_skills(text),
    }

# Load CSV dataset for training
df = pd.read_csv("Resume.csv", encoding="utf-8")
df = df.loc[df["Category"] == "INFORMATION-TECHNOLOGY"]

# Apply format checking to dataset
df["Proper_Format"] = df["Resume_html"].apply(check_format).astype(int)

# Convert resume text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["Resume_html"]).toarray()
y = df["Proper_Format"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to check if a new resume is properly formatted
def check_resume_format(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    vectorized_text = vectorizer.transform([text]).toarray()
    is_valid = model.predict(vectorized_text)[0]
    return "Properly Formatted" if is_valid else "Not Properly Formatted"

# Test with a sample PDF
sample_pdf = "10089434.pdf"
result = check_resume_format(sample_pdf)
print(f"Resume Format Check Result: {result}")
'''
df = pandas.read_csv("Resume.csv")
#parse csv training data
df = df.loc[df["Category"]=="INFORMATION-TECHNOLOGY"]
print(df)
for row in df.values:
     data = row[2]
     parsed_data = BeautifulSoup(data)
     print(parsed_data.find('div').text)
'''
