import streamlit as st
import nltk
import spacy
nltk.download('stopwords')
spacy.load('en_core_web_sm')
import pandas as pd
import base64, random
import time, datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, ds_jobs, web_jobs, android_jobs, ios_jobs, uiux_jobs
import pafy
import plotly.express as px
import youtube_dl
import re
import PyPDF2
import spacy
from spacy.matcher import Matcher
import streamlit as st
import PyPDF2
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
save_image_path = None
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO

def parse_pdf(file_path):
    resource_manager = PDFResourceManager()
    output_stream = StringIO()
    laparams = LAParams()

    with open(file_path, 'rb') as file:
        interpreter = PDFPageInterpreter(resource_manager, TextConverter(resource_manager, output_stream, laparams=laparams))
        for page in PDFPage.get_pages(file, check_extractable=True):
            interpreter.process_page(page)

    content = output_stream.getvalue()
    output_stream.close()

    return content
# load pre-trained model
nlp = spacy.load('en_core_web_sm')

def extract_resume_data(file_path):
    # Open pdf file
    pdfFileObj = open(file_path, 'rb')

    # Read file
    pdfReader = PyPDF2.PdfReader(pdfFileObj)

    # Get total number of pages
    num_pages = len(pdfReader.pages)

    # Initialize a count for the number of pages
    count = 0

    # Initialize an empty string variable
    text = ""

    # Extract text from every page in the file
    while count < num_pages:
        pageObj = pdfReader.pages[count]
        count += 1
        text += pageObj.extract_text()

    # Convert all strings to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Create dictionary with industrial and system engineering key terms by area
    
    terms = {
        'Data Science': [
    'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit',
    'data analysis', 'statistical modeling', 'r programming', 'data visualization', 'sql',
    'big data', 'natural language processing', 'predictive analytics', 'neural networks',
    'data mining', 'time series analysis', 'cloud computing', 'hadoop', 'spark',
    'bayesian statistics', 'a/b testing', 'feature engineering', 'dimensionality reduction',
    'unsupervised learning', 'supervised learning', 'reinforcement learning', 'data cleaning',
    'data wrangling', 'data engineering', 'data warehousing', 'etl', 'business intelligence',
    'decision trees', 'random forests', 'support vector machines', 'logistic regression',
    'linear regression', 'optimization', 'experiment design', 'ensemble methods', 'clustering',
    'data governance', 'data ethics', 'data privacy', 'data security', 'data strategy',
    'data integration', 'data science lifecycle'
],
        'Web Development': [
    'react', 'django', 'node.js', 'react.js', 'php', 'laravel', 'magento', 'wordpress',
    'javascript', 'angularjs', 'c#', 'flask', 'html', 'css', 'responsive design',
    'front-end development', 'back-end development', 'full-stack development', 'restful apis',
    'web security', 'database design', 'ui/ux design', 'version control', 'agile development',
    'cross-browser compatibility', 'performance optimization', 'testing and debugging',
    'web accessibility', 'seo', 'cms', 'e-commerce development', 'responsive frameworks',
    'web analytics', 'server administration', 'devops', 'continuous integration',
    'web scraping', 'web performance', 'web standards', 'scalability', 'graphql',
    'progressive web apps', 'microservices', 'serverless architecture', 'single page applications',
    'websockets', 'authentication and authorization', 'payment gateway integration',
    'mobile-first development', 'code optimization', 'c++', 'python', 'java'
],
        'Android Development': [
    'android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy', 'java',
    'android studio', 'material design', 'firebase', 'json', 'mvvm', 'restful apis',
    'sqlite', 'gradle', 'android jetpack', 'dependency injection', 'testing', 'google play store',
    'push notifications', 'background services', 'fragments', 'recyclerview', 'permissions handling',
    'localization', 'image and video processing', 'sensors integration', 'maps and location services',
    'bluetooth communication', 'wearable apps', 'in-app purchases', 'security and encryption',
    'app optimization', 'app publishing', 'app monetization', 'cross-platform development',
    'app widgets', 'deep linking', 'android ndk', 'performance profiling', 'memory management',
    'app notifications', 'android architecture components', 'unit testing', 'design patterns',
    'app updates and versioning', 'app store optimization', 'app security'
]
,
        'IOS Development': [
    'ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode', 'swiftui',
    'uikit', 'core data', 'auto layout', 'storyboards', 'interface builder',
    'core animation', 'restful apis', 'json', 'firebase', 'push notifications',
    'app store connect', 'localization', 'in-app purchases', 'core location',
    'mapkit', 'networking', 'core bluetooth', 'core graphics', 'core image',
    'core ml', 'sirikit', 'healthkit', 'arkit', 'core motion', 'core audio',
    'testflight', 'app security', 'app extensions', 'notifications and background modes',
    'app design guidelines', 'app publishing', 'code signing', 'provisioning profiles',
    'memory management', 'unit testing', 'design patterns', 'app lifecycle',
    'app updates and versioning', 'app performance optimization', 'widgetkit',
    'swift package manager', 'accessibility', 'app analytics'
],
        'UI/UX Development': [
    'ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
    'storyframes', 'adobe photoshop', 'editing', 'adobe illustrator', 'illustrator',
    'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro',
    'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 'user research',
    'user experience', 'visual design', 'typography', 'color theory', 'design systems',
    'responsive design', 'mobile design', 'icon design', 'illustration', 'motion graphics',
    'user flows', 'design thinking', 'user-centered design', 'persona development',
    'branding', 'usability testing', 'interaction design', 'information architecture',
    'user persona', 'card sorting', 'design critique', 'design collaboration',
    'data visualization', 'responsive web design', 'mobile app design', 'human-computer interaction',
    'cognitive psychology', 'user interface animation', 'information design', 'usability heuristics',
    'user testing', 'a/b testing', 'accessibility design', 'user engagement'
]
    }

    # Initialize score counters for each area
    dataScience = 0
    webDevelopment = 0
    androidDevelopment = 0
    iosDevelopment = 0
    uiux = 0

    # Create an empty list to store the scores
    scores = []

    # Obtain the scores for each area
    for area in terms.keys():
        if area == 'Data Science':
            for word in terms[area]:
                if word in text:
                    dataScience += 1
            scores.append(dataScience)
        elif area == 'Web Development':
            for word in terms[area]:
                if word in text:
                    webDevelopment += 1
            scores.append(webDevelopment)
        elif area == 'Android Development':
            for word in terms[area]:
                if word in text:
                    androidDevelopment += 1
            scores.append(androidDevelopment)
        elif area == 'IOS Development':
            for word in terms[area]:
                if word in text:
                    iosDevelopment += 1
            scores.append(iosDevelopment)
        else:
            for word in terms[area]:
                if word in text:
                    uiux += 1
            scores.append(uiux)
    total = sum(scores)
    finalScores = []
    for score in scores:
        temp = (score / total) * 100
        finalScores.append(round(temp, 2))

    # Create a summary dataframe
    summary = pd.DataFrame(finalScores, index=terms.keys(), columns=['score']).sort_values(by='score', ascending=False)

    # Close the pdf file
    pdfFileObj.close()

    return summary

def display_pie_chart(summary):
    # Create a pie chart
    pie = plt.figure(figsize=(10, 10))
    plt.pie(summary['score'], labels=summary.index, explode=(0.1, 0, 0, 0, 0), autopct='%1.0f%%', shadow=True,
            startangle=90)
    plt.axis('equal')

    # Adjust the position of labels to prevent overlapping
    plt.legend(title="Areas", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()

    # Display the pie chart in Streamlit
    st.pyplot(pie)

def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

#phone-number
def get_phone_numbers(string):
    r = re.compile(r'\b\d{10}\b')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', num) for num in phone_numbers]

#Name
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
    
#Skills
# load pre-trained model
nlp = spacy.load('en_core_web_sm')
def extract_skills(resume_text):
    nlp_text = nlp(resume_text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]
    
    skills = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit',
    'react', 'django', 'node.js', 'react.js', 'php', 'laravel', 'magento', 'wordpress',
    'android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy',
    'ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode',
    'ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
    'adobe photoshop', 'editing', 'adobe illustrator', 'illustrator', 'adobe after effects',
    'after effects', 'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign',
    'wireframe', 'solid', 'grasp', 'user research', 'user experience',
    'mysql', 'sql', 'computer vision', 'opencv', 'mongodb',
    'artificial intelligence', 'ai', 'data structures', 'python', 'c++', 'matlab',
    'css', 'html', 'github', 'data analysis', 'statistical modeling', 'r programming',
    'data visualization', 'big data', 'natural language processing', 'predictive analytics',
    'neural networks', 'data mining', 'time series analysis', 'cloud computing', 'hadoop',
    'spark', 'bayesian statistics', 'a/b testing', 'feature engineering', 'dimensionality reduction',
    'unsupervised learning', 'supervised learning', 'reinforcement learning', 'data cleaning',
    'data wrangling', 'data engineering', 'data warehousing', 'etl', 'business intelligence',
    'decision trees', 'random forests', 'support vector machines', 'logistic regression',
    'linear regression', 'optimization', 'experiment design', 'ensemble methods', 'clustering',
    'data governance', 'data ethics', 'data privacy', 'data security', 'data strategy',
    'data integration', 'data science lifecycle']
    
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

#College
def extract_college(text):
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
      return list(institute_names)  # Convert set back to a list

#Projects
def extract_projects(text):
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{'LOWER': 'developed'}],
        [{'LOWER': 'built'}],
        [{'LOWER': 'implemented'}]
    ]
    matcher.add("ProjectKeywords", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    project_keywords = []
    for match_id, start, end in matches:
        span = doc[start:end]
        project_keywords.append(span.text)
    
    return project_keywords

def get_hobbies(string):
    sub_patterns = ['[A-Z][a-z]* Singing', '[A-Z][a-z]* Dancing', r'\b(\w+\s+Hobbies)\b', r'\b(\w+\s+Interests)\b', 
                  r'\b(\w+\s+Gardening)\b',
                  'Stamp Collection [A-Z][a-z]*']
    hobby_names = set()  # Use a set to store unique institute names
    for pattern in sub_patterns:
      matches = re.findall(pattern, string)
      if matches:
          hobby_names.update(matches)
      if hobby_names:
        return list(hobby_names)

def get_resume_for_parsing(file):
    file_for_parsing = file
    num_pages = get_pdf_page_count(file_for_parsing)
    file_data = parse_pdf(file_for_parsing)
    text = file_data
    extracted_text = {}
    extracted_text["No of pages"] = num_pages
    email = get_email_addresses(text)
    #print(email)
    extracted_text["E-Mail"] = email
    phone_number= get_phone_numbers(text)
    #print(phone_number)
    extracted_text["Phone Number"] = phone_number
    name = extract_name(text)
    #print(name)
    extracted_text["Name"] = name
    #skills 
    skills = []
    skills = extract_skills(text)
    extracted_text["Skills"] = skills
    #college
    college = extract_college(text)
    extracted_text["Institutes"] = college
    projects = []
    projects = extract_projects(text)
    extracted_text["Projects"] = projects
    hobbies = []
    hobbies = get_hobbies(text)
    extracted_text["Hobbies"] = hobbies
    return extracted_text


def fetch_yt_video(link):
    video = pafy.new(link)
    return video.title

def get_pdf_page_count(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        page_count = len(pdf_reader.pages)
    return page_count


def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

def jobs_recommender(job_list):
    st.subheader("Job Recommendations:")
    for job in job_list:
        job_description, job_link = job
        st.markdown(f"[{job_description}]({job_link})")

st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon='/home/ec2-user/Resume-analysis-tool/Logo/SRA_Logo.ico',
)


def run():
    st.title("Resume parsing tool")
    choice = 'Normal User'
    img = Image.open('/home/ec2-user/Resume-analysis-tool/Logo/SRA_Logo.jpg')
    img = img.resize((250, 250))
    st.image(img)

    if choice == 'Normal User':
        # st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Upload your resume, and get smart recommendation based on it."</h4>''',
        #             unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            # with st.spinner('Uploading your Resume....'):
            #     time.sleep(4)
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = get_resume_for_parsing(save_image_path)
            summary = extract_resume_data(save_image_path)
            if resume_data:
                ## Get the whole resume data
                st.header("**Resume Analysis**")
                st.success("Hello " + resume_data['Name'])
                st.subheader("**Your Basic info**")
                st.text('Name: ' + resume_data['Name'])
                st.text(resume_data['E-Mail'])
                st.text(resume_data['Phone Number'])

                cand_level = ''
                if resume_data['No of pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',
                                unsafe_allow_html=True)
                elif resume_data['No of pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',
                                unsafe_allow_html=True)
                elif resume_data['No of pages'] >= 3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',
                                unsafe_allow_html=True)
                    
                st.subheader("**Skills Recommendation💡**")
                ## Skill shows
                keywords = st_tags(label='### Skills that you have',
                                   value=resume_data['Skills'])

                ##  recommendation
                ds_keyword = [
    'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit',
    'data analysis', 'statistical modeling', 'r programming', 'data visualization', 'sql',
    'big data', 'natural language processing', 'predictive analytics', 'neural networks',
    'data mining', 'time series analysis', 'cloud computing', 'hadoop', 'spark',
    'bayesian statistics', 'a/b testing', 'feature engineering', 'dimensionality reduction',
    'unsupervised learning', 'supervised learning', 'reinforcement learning', 'data cleaning',
    'data wrangling', 'data engineering', 'data warehousing', 'etl', 'business intelligence',
    'decision trees', 'random forests', 'support vector machines', 'logistic regression',
    'linear regression', 'optimization', 'experiment design', 'ensemble methods', 'clustering',
    'data governance', 'data ethics', 'data privacy', 'data security', 'data strategy',
    'data integration', 'data science lifecycle'
]

                web_keyword = [
    'react', 'django', 'node.js', 'react.js', 'php', 'laravel', 'magento', 'wordpress',
    'javascript', 'angularjs', 'c#', 'flask', 'html', 'css', 'responsive design',
    'front-end development', 'back-end development', 'full-stack development', 'restful apis',
    'web security', 'database design', 'ui/ux design', 'version control', 'agile development',
    'cross-browser compatibility', 'performance optimization', 'testing and debugging',
    'web accessibility', 'seo', 'cms', 'e-commerce development', 'responsive frameworks',
    'web analytics', 'server administration', 'devops', 'continuous integration',
    'web scraping', 'web performance', 'web standards', 'scalability', 'graphql',
    'progressive web apps', 'microservices', 'serverless architecture', 'single page applications',
    'websockets', 'authentication and authorization', 'payment gateway integration',
    'mobile-first development', 'code optimization', 'c++', 'python', 'java'
]
                android_keyword = [
    'android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy', 'java',
    'android studio', 'material design', 'firebase', 'json', 'mvvm', 'restful apis',
    'sqlite', 'gradle', 'android jetpack', 'dependency injection', 'testing', 'google play store',
    'push notifications', 'background services', 'fragments', 'recyclerview', 'permissions handling',
    'localization', 'image and video processing', 'sensors integration', 'maps and location services',
    'bluetooth communication', 'wearable apps', 'in-app purchases', 'security and encryption',
    'app optimization', 'app publishing', 'app monetization', 'cross-platform development',
    'app widgets', 'deep linking', 'android ndk', 'performance profiling', 'memory management',
    'app notifications', 'android architecture components', 'unit testing', 'design patterns',
    'app updates and versioning', 'app store optimization', 'app security'
]

                ios_keyword = [
    'ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode', 'swiftui',
    'uikit', 'core data', 'auto layout', 'storyboards', 'interface builder',
    'core animation', 'restful apis', 'json', 'firebase', 'push notifications',
    'app store connect', 'localization', 'in-app purchases', 'core location',
    'mapkit', 'networking', 'core bluetooth', 'core graphics', 'core image',
    'core ml', 'sirikit', 'healthkit', 'arkit', 'core motion', 'core audio',
    'testflight', 'app security', 'app extensions', 'notifications and background modes',
    'app design guidelines', 'app publishing', 'code signing', 'provisioning profiles',
    'memory management', 'unit testing', 'design patterns', 'app lifecycle',
    'app updates and versioning', 'app performance optimization', 'widgetkit',
    'swift package manager', 'accessibility', 'app analytics'
]
                uiux_keyword =  [
    'ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
    'storyframes', 'adobe photoshop', 'editing', 'adobe illustrator', 'illustrator',
    'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro',
    'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 'user research',
    'user experience', 'visual design', 'typography', 'color theory', 'design systems',
    'responsive design', 'mobile design', 'icon design', 'illustration', 'motion graphics',
    'user flows', 'design thinking', 'user-centered design', 'persona development',
    'branding', 'usability testing', 'interaction design', 'information architecture',
    'user persona', 'card sorting', 'design critique', 'design collaboration',
    'data visualization', 'responsive web design', 'mobile app design', 'human-computer interaction',
    'cognitive psychology', 'user interface animation', 'information design', 'usability heuristics',
    'user testing', 'a/b testing', 'accessibility design', 'user engagement'
]
                recommended_skills = []
                reco_field = ''
                rec_course = ''
                rec_job = ''
                ## Courses recommendation
                for i in resume_data['Skills']:
                    ## Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                                              'Data Mining', 'Clustering & Classification', 'Data Analytics',
                                              'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                                              'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                                              'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='2')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        rec_job = jobs_recommender(ds_jobs)
                        break

                    ## Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                                              'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='3')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)
                        rec_job = jobs_recommender(web_jobs)
                        break

                    ## Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                                              'Kivy', 'GIT', 'SDK', 'SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='4')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        rec_job = jobs_recommender(android_jobs)
                        break

                    ## IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                                              'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                                              'Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='5')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        rec_job = jobs_recommender(ios_jobs)
                        break

                    ## Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                                              'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing',
                                              'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe',
                                              'Solid', 'Grasp', 'User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='6')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        rec_job = jobs_recommender(uiux_jobs)
                        break

                #
                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date + '_' + cur_time)

                ### Resume writing recommendation
                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0
                if 'E-Mail' in resume_data:
                    resume_score = resume_score + 25
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Email.</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add your career Email, it will give your contact information to the Recruiters.</h4>''',
                        unsafe_allow_html=True)

                if len(resume_data["Skills"]) > 5 in resume_data:
                    resume_score = resume_score + 25
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Aall Skills/h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add as many skills as you know. It will give the assurance that everything written on your resume describes you fully</h4>''',
                        unsafe_allow_html=True)

                if 'Hobbies' in resume_data:
                    resume_score = resume_score + 25
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies⚽</h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Hobbies⚽. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',
                        unsafe_allow_html=True)

                if 'Projects' in resume_data:
                    resume_score = resume_score + 25
                    st.markdown(
                        '''<h4 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻 </h4>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h4 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Projects👨‍💻. It will show that you have done work related the required position or not.</h4>''',
                        unsafe_allow_html=True)

                st.subheader("**Resume Score📝**")
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score += 1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                st.success('** Your Resume Writing Score: ' + str(score) + '**')
                st.warning(
                    "** Note: This score is calculated based on the content that you have added in your Resume. **")
                st.title("Job score Analysis")
                st.subheader("Resume Summary")
                st.dataframe(summary)
                st.subheader("Resume Decomposition by Areas")
                display_pie_chart(summary)
                st.balloons()
            else:
                st.error('Something went wrong..')


run()
