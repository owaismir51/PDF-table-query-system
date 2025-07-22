# Intelligent PDF Data Extraction & Query System

# Project Overview
This project implements a robust and intelligent system for extracting structured, semi-structured, and unstructured data from PDF documents, transforming it into queryable database tables. It is specifically designed to handle complex PDF layouts often found in official reports like Annual Reports or Sustainability Reports, aiming for a fully automated data extraction process.

The system features a Streamlit-based user interface for easy interaction, backend data processing using a cascade of Python libraries, and integrates with a Large Language Model (LLM) for natural language querying capabilities.

# Features
PDF Input: Accepts PDF documents (both text-based and scanned) through an intuitive file uploader.
Advanced Data Extraction Pipeline:
Table Extraction: Employs a multi-strategy approach for table detection and extraction:
Utilizes pdfplumber for native PDF table recognition.
Leverages tabula-py for advanced table extraction, intelligently trying both stream and lattice modes with automatic table area guessing to maximize success across diverse table layouts. Robust error handling is implemented to prevent crashes from problematic PDF structures (e.g., JavaTabulaError due to non-orthogonal lines).
Key-Value Pair Extraction: Identifies and extracts key-value information (e.g., "Total Assets: $X") using regular expression patterns.
Critical Value Detection: Extracts specific, important data points (e.g., financial totals, dates, invoice numbers) from the document.
Optical Character Recognition (OCR): Integrates pytesseract to perform OCR on image-based content within PDFs, ensuring data from scanned documents or embedded images is also extracted.
Dynamic Data Structuring & Storage:
All extracted data (tables, key-value pairs, critical values) are transformed into pandas DataFrames.
These DataFrames are then dynamically loaded into an in-memory SQLite database, creating a structured and immediately queryable representation of the PDF's content. The database state is automatically reset for each new PDF upload, ensuring clean processing.
Intelligent Data Querying:
Natural Language Interface: Allows users to ask questions about the extracted data in plain English (e.g., "What are the company's revenues for 2022?").
LLM Integration (Groq): Leverages a powerful Large Language Model (LLM) through the Groq API to convert natural language queries into executable SQL and MongoDB queries. The system is configured to use an LLM with a large context window (e.g., llama-3.1-8b-instant or llama-3.1-70b-versatile) to handle comprehensive schema inputs.
Direct SQL Execution: Provides an option for users to execute custom SQL queries directly against the extracted data in the SQLite database.
User-Friendly Output:
Displays extracted tables, key-value pairs, and critical data previews within the Streamlit application.
Offers a convenient direct download option for all extracted tables as a single, combined CSV file, eliminating the need for local file saving and streamlining user experience.

# Technologies Used
Frontend & Application Framework:
Streamlit: For building the interactive web application.
PDF Processing Libraries (Python):
pdfplumber: For parsing PDFs, extracting text, and native table detection.
tabula-py: For advanced PDF table extraction (Python wrapper for Tabula Java library).
PyTesseract: Python wrapper for Google's Tesseract-OCR Engine for image-to-text conversion.
Pillow (PIL Fork): Image processing library, a dependency for pytesseract.
Java Runtime Environment (JRE/JDK) & JAI:
Java Development Kit (JDK) 8 or higher: Required for tabula-py to run.
Java Advanced Imaging (JAI) Image I/O Tools: Specific Java libraries needed by tabula-py's underlying PDFBox to correctly process JPEG2000 images, crucial for handling image-heavy PDFs.
Data Handling & Database:
pandas: For data manipulation and working with DataFrames.
SQLite3: Python's built-in lightweight SQL database for temporary storage and querying of extracted data.
LLM Integration:
Groq API: Provides high-performance inference for Large Language Models (LLMs) used for natural language query generation.
Specific LLM Model: Configured to use a Groq model with a large context window (e.g., llama-3.1-8b-instant or llama-3.1-70b-versatile).
Utilities:
python-dotenv: For loading environment variables (e.g., API keys) from a .env file.
pyperclip: For cross-platform copy-to-clipboard functionality.

# Setup Instructions
Follow these steps to set up and run the project locally.

# 1. Prerequisites
Python 3.8+: Download and install from python.org.
Java Development Kit (JDK): JDK 8 or higher is required for tabula-py.
Download from Oracle JDK or OpenJDK.
Ensure JAVA_HOME environment variable is set to your JDK installation directory (e.g., C:\Program Files\jdk-24.0.1).
Ensure %JAVA_HOME%\bin is added to your system's Path environment variable.
Verify installation: Open a new command prompt and run java -version and javac -version.
Tesseract-OCR Engine: Required for OCR capabilities.
Download the installer for Windows from Tesseract-OCR for Windows (look for an executable installer).
During installation, ensure you select the option to add Tesseract to your system's PATH.
Verify installation: Open a new command prompt and run tesseract --version.
Java Advanced Imaging (JAI) Image I/O Tools: Necessary for tabula-py to process JPEG2000 images in PDFs.
Download: Search for "jai_imageio-1_0_01-lib-windows-i586.exe" from Oracle's archive (it might require an Oracle account).
Manual Extraction: The .exe installer might fail on newer Java versions. Use an archive tool (like 7-Zip) to extract its contents.
Locate JARs: Navigate into the extracted folder (e.g., jai_imageio-1_0_01-lib-windows-i586\lib). You should find jai_imageio.jar and clibwrapper_jiio.jar.
Copy JARs: Create a dedicated folder (e.g., C:\Java_Jars_for_Tabula) and copy these .jar files into it.

# 2. Project Setup
Clone the repository (or download the project files):
git clone <your-repository-url>
cd pdf_to_sql
Create a Python Virtual Environment (recommended):
python -m venv venv
Activate the Virtual Environment:
On Windows:
.\venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate
Install Required Python Packages:
Create a requirements.txt file in your project root (pdf_to_sql/requirements.txt) with the content provided in the "Revised requirements.txt" section above.
Then, install:
pip install -r requirements.txt

# 3. Configuration
Groq API Key:
Go to Groq Console and generate an API key.
Create a file named .env in the root directory of your project (pdf_to_sql/.env).
Add your API key to the .env file:
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="llama-3.1-8b-instant" # Or "llama-3.1-70b-versatile" for larger models
Important: Ensure no extra spaces around = and that llama-3.1-8b-instant (or your chosen model) is the selected model, as mixtral-8x7b-32768 has been decommissioned.
Update JAI JAR Paths in table_processor.py:

Open table_processor.py.
Locate the _start_jvm method.
Update the jai_jars list with the absolute paths to the jai_imageio.jar and clibwrapper_jiio.jar files you copied to C:\Java_Jars_for_Tabula (or your chosen folder).
# In table_processor.py, within the _start_jvm method:
jai_jars = [
    r"C:\Java_Jars_for_Tabula\jai_imageio.jar", # Make sure this path is correct
    r"C:\Java_Jars_for_Tabula\clibwrapper_jiio.jar", # Make sure this path is correct
]
Save table_processor.py.

# 4. Run the Application
Open a NEW terminal (to ensure all environment variables are loaded).
Activate your virtual environment (if you closed the previous terminal).
Navigate to your project directory:
cd C:\Users\shaht\OneDrive\Desktop\Gen_AI\pdf_to_sql
Run the Streamlit application:
streamlit run app.py
The application will open in your web browser. You can then upload a PDF document and interact with the system.
