import tika
tika.initVM()
from tika import parser
import re

file = "/Users/sosarkar/collegeprojects/Smart_Resume_Analyser_App/Uploaded_Resumes/Somdyuti-Sarkar-Resume.pdf"
file_data = parser.from_file(file)
text = file_data['content']
extracted_text = {}
#E-MAIL

def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

email = get_email_addresses(text)
#print(email)
extracted_text["E-Mail"] = email


def get_phone_numbers(string):
    r = re.compile(r'\b\d{10}\b')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', num) for num in phone_numbers]

phone_number= get_phone_numbers(text)
#print(phone_number)
extracted_text["Phone Number"] = phone_number
#Name
import spacy
from spacy.matcher import Matcher

# load pre-trained model
nlp = spacy.load('en_core_web_sm')

# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

def extract_name(resume_text):
    nlp_text = nlp(resume_text)
    
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    
    matcher.add('NAME', [pattern], on_match = None)
    
    matches = matcher(nlp_text)
    
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
    
    
name = extract_name(text)
#print(name)
extracted_text["Name"] = name
import spacy

# load pre-trained model
nlp = spacy.load('en_core_web_sm')


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    skills = ["machine learning",
             "deep learning",
             "nlp",
             "natural language processing",
             "mysql",
             "sql",
             "django",
             "computer vision",
              "tensorflow",
             "opencv",
             "mongodb",
             "artificial intelligence",
             "ai",
             "flask",
             "robotics",
             "data structures",
             "python",
             "c++",
             "matlab",
             "css",
             "html",
             "github",
             "php"]
    
    skillset = []
    
    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)
    
    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

skills = []
skills = extract_skills(text)

extracted_text["Skills"] = skills
#skills 
import re

sub_patterns = ['[A-Z][a-z]* University', '[A-Z][a-z]* Educational Institute',
                'University of [A-Z][a-z]*', r'\b(\w+\s+College of Engineering)\b', 
                r'\b(\w+\s+University)\b',
                'Ecole [A-Z][a-z]*']
institute_names = set()  # Use a set to store unique institute names

for pattern in sub_patterns:
    matches = re.findall(pattern, text)
    if matches:
        institute_names.update(matches)

if institute_names:
    extracted_text["Institutes"] = list(institute_names)  # Convert set back to a list

print(extracted_text)