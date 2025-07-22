# import streamlit as st
# import os
# from dotenv import load_dotenv
# from groq import Groq
# import json
# import pyperclip
# import pandas as pd
# from table_processor import TableProcessor

# load_dotenv()
# API_KEY = os.getenv("GROQ_API_KEY")
# MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
# client = Groq(api_key=API_KEY)

# def generate_queries(prompt: str, schema: str) -> dict:
#     system_prompt = f"""
# You are a database expert.
# Generate valid MongoDB and SQL queries based on this schema:
# {schema}

# Rules:
# - Always return valid JSON.
# - For MongoDB regex use: {{"field": {{"$regex": "pattern", "$options": "i"}}}}
# - SQL must be standard.
# - Include explanation.

# Example:
# {{
#   "mongodb_query": {{ }},
#   "sql_query": "...",
#   "explanation": "..."
# }}
# """

#     try:
#         response = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt}
#             ],
#             model=MODEL,
#             temperature=0.3,
#             response_format={"type": "json_object"}
#         )
#         result = json.loads(response.choices[0].message.content)
#         json.dumps(result)  # Validate
#         return result
#     except Exception as e:
#         return {"error": str(e)}

# def main():
#     st.set_page_config(page_title="PDF Data Extraction & Query System", layout="wide", page_icon="📊")
#     st.title("📊 PDF Data Extraction & Intelligent Query System")

#     uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
#     max_pages = st.slider("Max pages to scan", min_value=1, max_value=30, value=5)

#     if uploaded_file:
#         tp = TableProcessor()

#         with st.spinner("Extracting structured tables..."):
#             tp.detect_tables_from_pdf(uploaded_file, max_pages=max_pages)

#         with st.spinner("Extracting key-value pairs..."):
#             tp.detect_key_value_pairs(uploaded_file, max_pages=max_pages)

#         with st.spinner("Scanning for critical values..."):
#             tp.detect_critical_values(uploaded_file)

#         tp.infer_schemas()

#         if not tp.tables and not tp.kv_pairs and not tp.critical_data:
#             st.warning("No structured data detected in the document.")
#             return

#         st.success(f"Extracted {len(tp.tables)} tables, {len(tp.kv_pairs)} key-value pairs, and {len(tp.critical_data)} critical values.")

#         if tp.tables:
#             st.subheader("📋 Extracted Tables")
#             table_idx = st.number_input("Select Table to Preview", min_value=0, max_value=len(tp.tables)-1, value=0)
#             df_preview = tp.get_table_preview(table_idx)
#             st.dataframe(df_preview)

#         if tp.kv_pairs:
#             st.subheader("🗒️ Extracted Key-Value Pairs")
#             df_kv = pd.DataFrame(tp.kv_pairs)
#             st.dataframe(df_kv)

#         if tp.critical_data:
#             st.subheader("🔍 Detected Critical Data")
#             df_crit = pd.DataFrame(list(tp.critical_data.items()), columns=["Key", "Value"])
#             st.dataframe(df_crit)

#         st.divider()
#         st.subheader("💬 Query the Extracted Data")
#         query_prompt = st.text_area("Describe the query you want:", height=100)

#         if st.button("Generate Queries"):
#             with st.spinner("Generating queries with LLM..."):
#                 schema_json = tp.export_schema()
#                 result = generate_queries(query_prompt, schema_json)

#                 if "error" in result:
#                     st.error(result["error"])
#                 else:
#                     st.session_state["last_mongo_query"] = result.get("mongodb_query", {})
#                     st.session_state["last_sql_query"] = result.get("sql_query", "")
#                     st.session_state["explanation"] = result.get("explanation", "")

#         if "last_mongo_query" in st.session_state:
#             st.divider()
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.subheader("MongoDB Query")
#                 st.code(json.dumps(st.session_state["last_mongo_query"], indent=2), language="json")
#                 if st.button("Copy MongoDB Query"):
#                     pyperclip.copy(json.dumps(st.session_state["last_mongo_query"], indent=2))
#                     st.toast("MongoDB Query Copied")
#             with col2:
#                 st.subheader("SQL Query")
#                 st.code(st.session_state["last_sql_query"], language="sql")
#                 if st.button("Copy SQL Query"):
#                     pyperclip.copy(st.session_state["last_sql_query"])
#                     st.toast("SQL Query Copied")

#             with st.expander("Explanation"):
#                 st.write(st.session_state["explanation"])

# if __name__ == "__main__":
#     main()

import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import json
import pyperclip
import pandas as pd
from table_processor import TableProcessor
import pytesseract
from PIL import Image
import jpype
from datetime import datetime # Import datetime for file naming
import io # Import io for BytesIO or StringIO

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
# IMPORTANT: Changed MODEL to one with larger context window
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant") 
client = Groq(api_key=API_KEY)

def generate_queries(prompt: str, schema: str) -> dict:
    system_prompt = f"""
You are a database expert.
Generate valid MongoDB and SQL queries based on this schema:
{schema}

Rules:
- Always return valid JSON.
- For MongoDB regex use: {{"field": {{"$regex": "pattern", "$options": "i"}}}}
- SQL must be standard.
- Include explanation.

Example:
{{
  "mongodb_query": {{ }},
  "sql_query": "...",
  "explanation": "..."
}}
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model=MODEL,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        json.dumps(result)  # Validate
        return result
    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(page_title="PDF Data Extraction & Query System", layout="wide", page_icon="📊")
    st.title("📊 PDF Data Extraction & Intelligent Query System")

    if 'tp' not in st.session_state:
        st.session_state.tp = TableProcessor()

    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    max_pages = st.slider("Max pages to scan", min_value=1, max_value=500, value=5) # Increased max pages for reports

    if uploaded_file:
        tp = st.session_state.tp

        try:
            with st.spinner("Extracting structured tables..."):
                tp.detect_tables_from_pdf(uploaded_file, max_pages=max_pages)

            with st.spinner("Extracting key-value pairs..."):
                tp.detect_key_value_pairs(uploaded_file, max_pages=max_pages)

            with st.spinner("Scanning for critical values..."):
                tp.detect_critical_values(uploaded_file)
            
            # Infer schemas for all extracted data (tables, KV pairs, critical values)
            tp.infer_schemas()

            if not tp.tables and not tp.kv_pairs and not tp.critical_data:
                st.warning("No structured data detected in the document.")
                return

            st.success(f"Extracted {len(tp.tables)} tables, {len(tp.kv_pairs)} key-value pairs, and {len(tp.critical_data)} critical values.")

            # --- Start: Combined CSV Download Section ---
            if tp.schemas: # Only show download option if there are tables in schemas
                st.subheader("📥 Download All Extracted Tables")
                
                combined_csv_buffers = [] # To hold StringIO objects for each table
                download_available = False

                for i, schema in enumerate(tp.schemas):
                    try:
                        df_to_save = pd.read_sql_query(f'SELECT * FROM "{schema["table_name"]}"', tp.conn)
                        
                        if not df_to_save.empty:
                            # Add a header for this table within the combined CSV
                            table_header_name = tp._sanitize_filename(schema['table_name'])
                            combined_csv_buffers.append(f"\n--- Table: {table_header_name} (Page {schema.get('source_page', 'N/A')}) ---\n")
                            
                            # Convert DataFrame to CSV string in memory
                            csv_buffer_single_table = io.StringIO()
                            df_to_save.to_csv(csv_buffer_single_table, index=False)
                            combined_csv_buffers.append(csv_buffer_single_table.getvalue())
                            combined_csv_buffers.append("\n") # Add a newline for separation
                            download_available = True
                        else:
                            st.info(f"Table '{schema['table_name']}' is empty and will not be included in the combined CSV download.")
                    except Exception as e_sql:
                        st.warning(f"Error preparing table '{schema['table_name']}' for combined download: {e_sql}")

                if download_available:
                    combined_csv_output = "".join(combined_csv_buffers)
                    download_filename = f"extracted_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="💾 Download All Tables as Single CSV",
                        data=combined_csv_output,
                        file_name=download_filename,
                        mime="text/csv",
                        key="download_all_tables_button" # Unique key
                    )
                    st.info("Combined CSV for download has been prepared.")
                else:
                    st.warning("No valid data found across all tables to create a downloadable CSV.")
            # --- End: Combined CSV Download Section ---

            if tp.tables:
                st.subheader("📋 Extracted Tables Preview")
                table_max_idx = len(tp.tables) - 1 if len(tp.tables) > 0 else 0
                if table_max_idx >= 0: # Ensure there's at least one table to select
                    table_idx = st.number_input("Select Table to Preview", min_value=0, max_value=table_max_idx, value=0, key="table_preview_slider")
                    df_preview = tp.get_table_preview(table_idx)
                    st.dataframe(df_preview)
                    
                    csv_filename_individual = st.text_input("Filename for this table (without extension)", value=f"table_{table_idx}", key=f"csv_filename_individual_{table_idx}")
                    if st.button(f"💾 Save Table {table_idx} as CSV (Local)", key=f"save_table_local_{table_idx}"):
                        try:
                            filepath = tp.save_table_to_csv(table_idx, f"{csv_filename_individual}.csv")
                            st.success(f"Table saved locally to: {filepath}")
                        except ValueError as ve:
                            st.error(f"Error saving table locally: {ve}")
                        except Exception as e:
                            st.error(f"Error saving table locally: {e}")
                else:
                    st.info("No tables available for preview.")


            if tp.kv_pairs:
                st.subheader("🗒️ Extracted Key-Value Pairs")
                df_kv = pd.DataFrame(tp.kv_pairs)
                st.dataframe(df_kv)

            if tp.critical_data:
                st.subheader("🔍 Detected Critical Data")
                df_crit = pd.DataFrame(list(tp.critical_data.items()), columns=["Key", "Value"])
                st.dataframe(df_crit)

            st.divider()
            st.subheader("💬 Query the Extracted Data")
            
            with st.expander("🔍 Run Direct SQL Query"):
                sql_query = st.text_area("Enter SQL Query:", height=100, key="direct_sql_query")
                if st.button("Execute SQL", key="execute_direct_sql"):
                    if not sql_query.strip():
                        st.warning("Please enter an SQL query to execute.")
                    else:
                        result = tp.execute_sql_query(sql_query)
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                        else:
                            st.info(result)

            query_prompt = st.text_area("Describe the query you want (natural language):", height=100, key="llm_query_prompt")

            if st.button("Generate Queries", key="generate_llm_queries"):
                if not query_prompt.strip():
                    st.warning("Please enter a query description.")
                else:
                    with st.spinner("Generating queries with LLM..."):
                        schema_json = tp.export_schema()
                        result = generate_queries(query_prompt, schema_json)

                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.session_state["last_mongo_query"] = result.get("mongodb_query", {})
                            st.session_state["last_sql_query"] = result.get("sql_query", "")
                            st.session_state["explanation"] = result.get("explanation", "")

            if "last_mongo_query" in st.session_state:
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("MongoDB Query")
                    st.code(json.dumps(st.session_state["last_mongo_query"], indent=2), language="json")
                    if st.button("Copy MongoDB Query", key="copy_mongo_query"):
                        pyperclip.copy(json.dumps(st.session_state["last_mongo_query"], indent=2))
                        st.toast("MongoDB Query Copied")
                with col2:
                    st.subheader("SQL Query")
                    st.code(st.session_state["last_sql_query"], language="sql")
                    if st.button("Copy SQL Query", key="copy_sql_query"):
                        pyperclip.copy(st.session_state["last_sql_query"])
                        st.toast("SQL Query Copied")
                    
                    if st.button("▶️ Execute This SQL", key="execute_llm_sql"):
                        if not st.session_state["last_sql_query"].strip():
                            st.warning("No SQL query generated to execute.")
                        else:
                            result = tp.execute_sql_query(st.session_state["last_sql_query"])
                            if isinstance(result, pd.DataFrame):
                                st.dataframe(result)
                            else:
                                st.info(result)

                with st.expander("Explanation"):
                    st.write(st.session_state["explanation"])

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
