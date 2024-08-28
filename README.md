# Interactive PDF Analysis Tool

![Demo Video](https://youtu.be/9i1_vJh-558)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Introduction

The Interactive PDF Analysis Tool is a powerful web application built using Streamlit that allows users to perform in-depth analysis of PDF documents. With features such as text extraction, word cloud generation, sentiment analysis, named entity recognition, and document comparison, this tool is designed for researchers, analysts, and anyone dealing with large volumes of PDF files.

## Features

- **PDF Text Extraction**: Extract and display text from uploaded PDF files.
- **Word Cloud Generation**: Visualize the most frequent words in the document.
- **Sentiment Analysis**: Analyze the sentiment (positive, negative, neutral) within the text.
- **Named Entity Recognition (NER)**: Identify and visualize entities such as names, organizations, locations, etc.
- **Document Comparison**: Compare multiple PDFs to find similarities and differences.
- **Text-to-Speech**: Convert the extracted text into speech.
- **Question Answering**: Ask questions about the content of the PDFs and get AI-generated answers.

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed on your machine.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/RonitGajbhiye358/Interactive-PDF-Analysis-Tool.git
   cd Interactive-PDF-Analysis-Tool
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your API keys and other necessary environment variables as follows:
     ```
     GOOGLE_API_KEY=your_google_api_key
     ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open your web browser and go to `http://localhost:8501` to start using the tool.

## Demo Video

![full functional demo](https://youtu.be/9i1_vJh-558)
*Demo video*

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Framework for creating interactive web apps.
- **PyPDF2**: Library for PDF text extraction.
- **Spacy**: Used for Named Entity Recognition (NER).
- **WordCloud**: For generating word clouds.
- **Pyttsx3**: Text-to-speech conversion.
- **LangChain**: For question-answering using AI.
- **FAISS**: For vector search and similarity comparison.
- **Matplotlib & Seaborn**: For plotting and visualizations.
- **Plotly**: For interactive charts and graphs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## Contact

**Ronit Gajbhiye**

- GitHub: [RonitGajbhiye358](https://github.com/RonitGajbhiye358)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/ronitgajbhiye/)

Feel free to contact me if you have any questions or suggestions!
