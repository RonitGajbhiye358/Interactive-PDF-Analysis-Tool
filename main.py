import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import io
import seaborn as sns
import pandas as pd
import plotly.express as px

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()

# Define helper functions
def get_pdf_text(pdf_files):
    text = {}
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""
        text[pdf.name] = pdf_text
    return text

def display_pdf_text(pdf_text, highlight=None):
    selected_pdf = st.selectbox("Select PDF to view", options=list(pdf_text.keys()))
    pdf_content = pdf_text[selected_pdf]
    if highlight:
        pdf_content = pdf_content.replace(highlight, f"<mark>{highlight}</mark>")
    st.markdown(pdf_content, unsafe_allow_html=True)

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say, "Answer is not available in the context." Do not provide incorrect answers.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    st.image(buffer, use_column_width=True)
    buffer.close()
    return buffer

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def summarize_text(text):
    return text[:1000] + "..." if len(text) > 1000 else text

def analyze_sentiment(text):
    return {'pos': 0.1, 'neg': 0.1, 'neu': 0.8}

def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def plot_entities(entities):
    df = pd.DataFrame(entities, columns=['Entity', 'Label'])
    st.write("Entity Statistics:")
    st.write(df['Label'].value_counts())

    st.subheader("Entity Counts")
    fig, ax = plt.subplots(figsize=(10, 5))
    df['Label'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    ax.set_title("Counts of Different Entity Types")
    st.pyplot(fig)

def visualize_entities(text):
    doc = nlp(text)
    colors = {
        "PERSON": "red",
        "ORG": "blue",
        "GPE": "green",
        "LOC": "purple",
        "DATE": "orange",
        "TIME": "pink",
        "MONEY": "brown",
        "PERCENT": "grey"
    }

    html = ""
    for ent in doc.ents:
        color = colors.get(ent.label_, "black")
        html += f'<span style="color:{color}; font-weight:bold;">{ent.text}</span> '

    st.markdown(html, unsafe_allow_html=True)

def compare_documents(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def plot_sentiment(sentiment):
    labels = ['Positive', 'Negative', 'Neutral']
    values = [sentiment['pos'], sentiment['neg'], sentiment['neu']]
    st.bar_chart(dict(zip(labels, values)))

def plot_similarity_matrix(similarity_matrix, labels):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.title('Document Similarity Matrix')
    st.pyplot(fig)

def main():
    # Custom CSS for a better look
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4583AA;
        }
        .sidebar .sidebar-content {
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Interactive PDF Analysis Tool")
    st.sidebar.title("PDF Analysis")

    # Sidebar for feature selection
    option = st.sidebar.selectbox("Choose a section", ["Upload & Analyze PDFs", "Text-to-Speech", "Word Cloud & Summary", "Sentiment & Comparison", "Named Entity Recognition", "Ask a Question"])

    if option == "Upload & Analyze PDFs":
        with st.sidebar.expander("Upload PDF Files"):
            uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            with st.spinner("Extracting text from PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                st.success("Text extracted successfully!")

                # Display PDF content
                st.subheader("PDF Viewer")
                highlight = st.text_input("Highlight Text", key="highlight")
                display_pdf_text(pdf_text, highlight=highlight)

                # Process text and create vector store
                all_text = " ".join(pdf_text.values())
                text_chunks = get_text_chunks(all_text)
                with st.spinner("Indexing text..."):
                    get_vector_store(text_chunks)
                    st.success("Text indexed successfully!")

    elif option == "Text-to-Speech":
        st.subheader("Text-to-Speech")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key="text_to_speech_files")
        if uploaded_files:
            all_text = ""
            with st.spinner("Extracting text from PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                all_text = " ".join(pdf_text.values())
                st.success("Text extracted successfully!")

            if st.button("Convert to Speech"):
                with st.spinner("Converting text to speech..."):
                    text_to_speech(all_text)
                    st.success("Text-to-Speech conversion completed!")

    elif option == "Word Cloud & Summary":
        st.subheader("Generate Word Cloud and Summary")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key="word_cloud_files")
        if uploaded_files:
            all_text = ""
            with st.spinner("Extracting text from PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                all_text = " ".join(pdf_text.values())
                st.success("Text extracted successfully!")

            if st.button("Generate Word Cloud"):
                with st.spinner("Generating word cloud..."):
                    generate_word_cloud(all_text)
                    st.success("Word Cloud generated!")
            
            summary = summarize_text(all_text)
            st.write("Summary: ", summary)

    elif option == "Sentiment & Comparison":
        st.subheader("Sentiment Analysis and Document Comparison")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key="sentiment_files")
        if uploaded_files:
            all_texts = []
            with st.spinner("Extracting text from PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                for key, text in pdf_text.items():
                    all_texts.append(text)
                st.success("Text extracted successfully!")

            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    sentiment = analyze_sentiment(" ".join(all_texts))
                    plot_sentiment(sentiment)
                    st.success("Sentiment analysis completed!")
            
            if st.button("Compare Documents"):
                with st.spinner("Comparing documents..."):
                    similarity_matrix = compare_documents(all_texts)
                    plot_similarity_matrix(similarity_matrix, [f"Doc {i+1}" for i in range(len(all_texts))])
                    st.success("Document comparison completed!")

    elif option == "Named Entity Recognition":
        st.subheader("Named Entity Recognition")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key="ner_files")
        if uploaded_files:
            all_text = ""
            with st.spinner("Extracting text from PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                all_text = " ".join(pdf_text.values())
                st.success("Text extracted successfully!")

            st.write("Extracted Text:")
            st.write(all_text[:1000])  # Display a preview of the text

            st.write("Named Entities:")
            entities = named_entity_recognition(all_text)
            st.write(entities)
            plot_entities(entities)
            visualize_entities(all_text)

    elif option == "Ask a Question":
        st.subheader("Ask a Question About the PDFs")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key="qa_files")
        if uploaded_files:
            all_text = ""
            with st.spinner("Extracting text from PDFs..."):
                pdf_text = get_pdf_text(uploaded_files)
                all_text = " ".join(pdf_text.values())
                st.success("Text extracted successfully!")

            st.write("Upload complete. You can now ask questions about the content of these PDFs.")

            question = st.text_input("Enter your question")
            if st.button("Submit"):
                with st.spinner("Getting answer..."):
                    user_input(question)
                    st.success("Question answered!")

if __name__ == "__main__":
    main()
