import streamlit as st
import re
import io

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


st.set_page_config(
    page_title="AI-Based Resume Screening and Job Matching System",
    page_icon="📄",
    layout="wide"
)

st.title("AI-Based Resume Screening and Job Matching System")

st.write(
    "Upload your resume and enter a job description to analyze the resume-job match."
)

SKILL_PATTERNS = {

    "python": [
        r"\bpython\b"
    ],

    "java": [
        r"(?<![a-zA-Z0-9])java(?![a-zA-Z0-9])"
    ],

    "javascript": [
        r"\bjavascript\b",
        r"(?<![a-zA-Z0-9])js(?![a-zA-Z0-9])"
    ],

    "c++": [
        r"(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])"
    ],

    "c#": [
        r"(?<![a-zA-Z0-9])c#(?![a-zA-Z0-9])",
        r"\bc\s*sharp\b"
    ],

    "html": [
        r"(?<![a-zA-Z0-9])html(?![a-zA-Z0-9])",
        r"\bhypertext\s+markup\s+language\b"
    ],

    "css": [
        r"(?<![a-zA-Z0-9])css(?![a-zA-Z0-9])",
        r"\bcascading\s+style\s+sheets?\b"
    ],

    "artificial intelligence": [
        r"\bartificial\s+intelligence\b",
        r"(?<![a-zA-Z0-9])ai(?![a-zA-Z0-9])"
    ],

    "machine learning": [
        r"\bmachine\s+learning\b",
        r"(?<![a-zA-Z0-9])ml(?![a-zA-Z0-9])"
    ],

    "deep learning": [
        r"\bdeep\s+learning\b",
        r"(?<![a-zA-Z0-9])dl(?![a-zA-Z0-9])"
    ],

    "natural language processing": [
        r"\bnatural\s+language\s+processing\b",
        r"(?<![a-zA-Z0-9])nlp(?![a-zA-Z0-9])"
    ],

    "computer vision": [
        r"\bcomputer\s+vision\b"
    ],

    "data analysis": [
        r"\bdata\s+analysis\b",
        r"\bdata\s+analytics\b"
    ],

    "data visualization": [
        r"\bdata\s+visualization\b",
        r"\bdata\s+visualisation\b"
    ],

    "statistics": [
        r"\bstatistics\b",
        r"\bstatistical\s+analysis\b"
    ],

    "data structures": [
        r"\bdata\s+structures?\b",
        r"(?<![a-zA-Z0-9])dsa(?![a-zA-Z0-9])"
    ],

    "algorithms": [
        r"\balgorithms?\b"
    ],

    "sql": [
        r"(?<![a-zA-Z0-9])sql(?![a-zA-Z0-9])",
        r"\bstructured\s+query\s+language\b"
    ],

    "mysql": [
        r"\bmysql\b"
    ],

    "mongodb": [
        r"\bmongodb\b",
        r"\bmongo\s*db\b"
    ],

    "dbms": [
        r"\bdbms\b",
        r"\bdatabase\s+management\s+system\b"
    ],

    "pandas": [
        r"\bpandas\b"
    ],

    "numpy": [
        r"\bnumpy\b"
    ],

    "scikit-learn": [
        r"\bscikit[-\s]?learn\b",
        r"\bsklearn\b"
    ],

    "tensorflow": [
        r"\btensorflow\b"
    ],

    "pytorch": [
        r"\bpytorch\b"
    ],

    "keras": [
        r"\bkeras\b"
    ],

    "nltk": [
        r"\bnltk\b",
        r"\bnatural\s+language\s+toolkit\b"
    ],

    "spacy": [
        r"\bspacy\b"
    ],

    "gensim": [
        r"\bgensim\b"
    ],

    "word2vec": [
        r"\bword2vec\b",
        r"\bword\s*2\s*vec\b"
    ],

    "tf-idf": [
        r"\btf[-\s]?idf\b",
        r"\btfidf\b",
        r"\bterm\s+frequency[-\s]+inverse\s+document\s+frequency\b"
    ],

    "cosine similarity": [
        r"\bcosine\s+similarity\b"
    ],

    "power bi": [
        r"\bpower\s*bi\b",
        r"\bpbi\b"
    ],

    "excel": [
        r"\bexcel\b",
        r"\bmicrosoft\s+excel\b",
        r"\bms\s+excel\b"
    ],

    "git": [
        r"(?<![a-zA-Z0-9])git(?![a-zA-Z0-9])"
    ],

    "github": [
        r"\bgithub\b"
    ],

    "jupyter": [
        r"\bjupyter\b",
        r"\bjupyter\s+notebook\b"
    ],

    "streamlit": [
        r"\bstreamlit\b"
    ],

    "flask": [
        r"\bflask\b"
    ],

    "django": [
        r"\bdjango\b"
    ],

    "operating systems": [
        r"\boperating\s+systems?\b"
    ],

    "computer networks": [
        r"\bcomputer\s+networks?\b",
        r"\bcomputer\s+networking\b"
    ],

    "aws": [
        r"(?<![a-zA-Z0-9])aws(?![a-zA-Z0-9])",
        r"\bamazon\s+web\s+services\b"
    ],

    "azure": [
        r"\bazure\b",
        r"\bmicrosoft\s+azure\b"
    ],

    "docker": [
        r"\bdocker\b"
    ],

    "linux": [
        r"\blinux\b"
    ]
}


def normalize_text(text):
    text = text.lower()
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_skills(text):
    text = normalize_text(text)
    found_skills = []

    for skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_skills.append(skill)
                break

    return found_skills


def calculate_tfidf(resume_text, job_text):
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(
        [
            resume_text,
            job_text
        ]
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return similarity * 100


def create_report(
    match_percentage,
    tfidf_score,
    skill_score,
    recommendation,
    required_skills,
    matched_skills,
    missing_skills
):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph(
            "Resume Screening Analysis Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            f"<b>Final Match Percentage:</b> {match_percentage:.2f}%",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>TF-IDF Similarity Score:</b> {tfidf_score:.2f}%",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Skill Matching Score:</b> {skill_score:.2f}%",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Match Recommendation",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            recommendation,
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Required Skills in Job Description",
            styles["Heading2"]
        )
    )

    for skill in required_skills:
        content.append(
            Paragraph(
                "• " + skill,
                styles["BodyText"]
            )
        )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Matched Skills",
            styles["Heading2"]
        )
    )

    if matched_skills:
        for skill in matched_skills:
            content.append(
                Paragraph(
                    "✓ " + skill,
                    styles["BodyText"]
                )
            )
    else:
        content.append(
            Paragraph(
                "No matched skills.",
                styles["BodyText"]
            )
        )

    content.append(
        Spacer(1, 15)
    )

    content.append(
        Paragraph(
            "Missing Skills",
            styles["Heading2"]
        )
    )

    if missing_skills:
        for skill in missing_skills:
            content.append(
                Paragraph(
                    "✗ " + skill,
                    styles["BodyText"]
                )
            )
    else:
        content.append(
            Paragraph(
                "No missing skills.",
                styles["BodyText"]
            )
        )

    document.build(content)
    buffer.seek(0)

    return buffer


uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)


if uploaded_file is not None:

    try:

        reader = PdfReader(
            uploaded_file
        )

        resume_text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                resume_text += extracted + "\n"

        if not resume_text.strip():

            st.error(
                "Unable to extract text from this PDF. "
                "Please upload a text-based PDF."
            )

        else:

            st.success(
                "Resume uploaded successfully!"
            )

            st.subheader(
                "Enter Job Description"
            )

            job_description = st.text_area(
                "Job Description",
                placeholder=(
                    "Python Developer – Looking for a Python "
                    "developer with skills in Python, AI, ML, "
                    "NLP, SQL and data analysis."
                ),
                height=150
            )

            if st.button(
                "Check Resume Match",
                type="primary"
            ):

                if not job_description.strip():

                    st.warning(
                        "Please enter a job description."
                    )

                else:

                    resume_skills = find_skills(
                        resume_text
                    )

                    job_skills = find_skills(
                        job_description
                    )

                    matched_skills = []

                    for skill in job_skills:

                        if skill in resume_skills:
                            matched_skills.append(skill)

                    missing_skills = []

                    for skill in job_skills:

                        if skill not in resume_skills:
                            missing_skills.append(skill)

                    tfidf_score = calculate_tfidf(
                        resume_text,
                        job_description
                    )

                    if len(job_skills) > 0:

                        skill_score = (
                            len(matched_skills)
                            /
                            len(job_skills)
                        ) * 100

                    else:

                        skill_score = 0

                    match_percentage = (
                        tfidf_score * 0.4
                        +
                        skill_score * 0.6
                    )

                    if match_percentage >= 75:

                        recommendation = (
                            "Excellent Match - The resume "
                            "strongly matches the job "
                            "requirements."
                        )

                    elif match_percentage >= 60:

                        recommendation = (
                            "Good Match - The resume matches "
                            "most of the job requirements."
                        )

                    elif match_percentage >= 40:

                        recommendation = (
                            "Moderate Match - The resume "
                            "matches some of the job "
                            "requirements."
                        )

                    else:

                        recommendation = (
                            "Low Match - The resume does not "
                            "match many of the required "
                            "job skills."
                        )

                    st.success(
                        "Resume and job description "
                        "received successfully!"
                    )

                    st.header(
                        "Resume Match Score"
                    )

                    st.progress(
                        min(
                            int(match_percentage),
                            100
                        )
                    )

                    st.write(
                        f"**Match Percentage: "
                        f"{match_percentage:.2f}%**"
                    )

                    st.write(
                        f"TF-IDF Similarity Score: "
                        f"{tfidf_score:.2f}%"
                    )

                    st.write(
                        f"Skill Matching Score: "
                        f"{skill_score:.2f}%"
                    )

                    st.subheader(
                        "Match Recommendation"
                    )

                    if match_percentage >= 75:

                        st.success(
                            "🟢 " + recommendation
                        )

                    elif match_percentage >= 40:

                        st.warning(
                            "🟡 " + recommendation
                        )

                    else:

                        st.error(
                            "🔴 " + recommendation
                        )

                    st.subheader(
                        "Required Skills in Job Description"
                    )

                    if job_skills:

                        for skill in job_skills:
                            st.write(
                                "• " + skill
                            )

                    else:

                        st.write(
                            "No recognized technical "
                            "skills found."
                        )

                    st.subheader(
                        "Matched Skills"
                    )

                    if matched_skills:

                        for skill in matched_skills:
                            st.write(
                                "✅ " + skill
                            )

                    else:

                        st.write(
                            "No matched skills found."
                        )

                    st.subheader(
                        "Missing Skills"
                    )

                    if missing_skills:

                        for skill in missing_skills:
                            st.write(
                                "❌ " + skill
                            )

                    else:

                        st.success(
                            "No missing skills."
                        )

                    st.subheader(
                        "Resume Improvement Suggestions"
                    )

                    if missing_skills:

                        st.write(
                            "Consider adding the following "
                            "skills to your resume if you "
                            "have knowledge or experience "
                            "in them:"
                        )

                        for skill in missing_skills:

                            st.write(
                                "💡 Consider adding: "
                                + skill
                            )

                    else:

                        st.success(
                            "Your resume contains all the "
                            "required skills."
                        )

                    st.header(
                        "Download Analysis Report"
                    )

                    pdf = create_report(
                        match_percentage,
                        tfidf_score,
                        skill_score,
                        recommendation,
                        job_skills,
                        matched_skills,
                        missing_skills
                    )

                    st.download_button(
                        label="📄 Download Report",
                        data=pdf,
                        file_name="Resume_Analysis_Report.pdf",
                        mime="application/pdf"
                    )

    except Exception as error:

        st.error(
            "An error occurred while processing "
            "the resume."
        )

        st.write(
            str(error)
        )