import re
import json
import os
import threading
import sqlite3
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import pdfplumber
from datetime import datetime
import pytesseract
from PIL import Image
import tabula
import jpype
import io
import tabula.errors

class TableProcessor:
    def __init__(self, storage_dir: str = "output_data"):
        self.tables: List[Dict] = []
        self.schemas: List[Dict] = []
        self.kv_pairs: List[Dict] = []
        self.critical_data: Dict[str, Any] = {}
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self._local = threading.local()
        self.db_path = os.path.join(self.storage_dir, "extracted_data.db")
        
        self._start_jvm()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[^\w\-_. ]', '_', name)

    def _start_jvm(self):
        try:
            if not jpype.isJVMStarted():
                jvm_path = r"C:\Program Files\jdk-24.0.1\bin\server\jvm.dll"
                
                jai_jars = [
                    r"C:\Java_Jars_for_Tabula\jai_imageio.jar",
                    r"C:\Java_Jars_for_Tabula\clibwrapper_jiio.jar",
                ]
                
                classpath_args = []
                for jar_path in jai_jars:
                    if not os.path.exists(jar_path):
                        print(f"Warning: JAI Image I/O Tools JAR not found at: {jar_path}. JPEG2000 images might not be processed.")
                    else:
                        classpath_args.append(jar_path)

                if not os.path.exists(jvm_path):
                    print(f"Warning: jvm.dll not found at: {jvm_path}. Please update the path in table_processor.py.")
                    raise FileNotFoundError(f"jvm.dll not found at: {jvm_path}")
                
                print("Starting JVM...")
                classpath_str = os.pathsep.join(classpath_args)
                
                jpype.startJVM(jvm_path, f"-Djava.class.path={classpath_str}", convertStrings=False)
                print("JVM Started.")
            else:
                print("JVM already running.")
        except Exception as e:
            print(f"Error starting JVM: {e}")
            raise

    def _reset_state(self):
        """Resets all extracted data and cleans up the SQLite database."""
        self.tables = []
        self.schemas = []
        self.kv_pairs = []
        self.critical_data = {}
        
        # Explicitly close and remove the SQLite database file
        try:
            if hasattr(self._local, "conn") and self._local.conn:
                self._local.conn.close()
                del self._local.conn # Remove the attribute so a new one is created next time
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                print(f"Cleaned up old SQLite database: {self.db_path}")
        except Exception as e:
            print(f"Warning: Could not fully clean up old SQLite database: {e}")

    def _process_extracted_dataframe(self, df_raw: pd.DataFrame, current_page_num: int, source_method: str) -> Optional[Dict]:
        """Helper to clean and validate a DataFrame extracted by tabula-py or pdfplumber."""
        df = df_raw.dropna(axis=0, how='all').dropna(axis=1, how='all')

        if df.empty:
            print(f"Skipping empty DataFrame from page {current_page_num} ({source_method}) after initial cleaning.")
            return None

        headers = []
        rows = []
        
        # Try to infer headers from first non-null row.
        first_valid_row_idx = None
        for row_idx in range(len(df)):
            if not df.iloc[row_idx].isnull().all():
                first_valid_row_idx = row_idx
                break
        
        if first_valid_row_idx is not None:
            headers = df.iloc[first_valid_row_idx].astype(str).str.strip().tolist()
            # Sanitize headers: remove special chars, replace spaces, ensure unique
            sanitized_headers = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '_')) for col in headers]
            final_headers = []
            for h_idx, h in enumerate(sanitized_headers):
                if not h: # If header is empty after sanitization
                    final_headers.append(f"Column_{h_idx}")
                elif h in final_headers: # If header is a duplicate
                    final_headers.append(f"{h}_{h_idx}")
                else:
                    final_headers.append(h)
            headers = final_headers
            
            rows = df.iloc[first_valid_row_idx + 1:].values.tolist()
        else: # If no valid header row is found (e.g., all first rows are empty), use generic headers
            headers = [f"col_{j+1}" for j in range(df.shape[1])]
            rows = df.values.tolist() # All rows are data

        # Further filter out rows that are entirely empty after processing
        processed_rows = []
        for r in rows:
            str_r = [str(x).strip() if x is not None else '' for x in r]
            if any(cell != '' for cell in str_r):
                # Ensure row length matches headers, pad/truncate as necessary
                if len(r) != len(headers):
                    padded_r = r + [None] * (len(headers) - len(r))
                    processed_rows.append(padded_r[:len(headers)])
                else:
                    processed_rows.append(r)
            else:
                print(f"Skipping empty row detected by cleaning in table from page {current_page_num} ({source_method}).")

        if headers and processed_rows and len(processed_rows) >= 1: # Require at least 1 data row
            return {
                'columns': headers,
                'rows': processed_rows,
                'source_page': current_page_num,
                'table_index': None # Will be assigned later
            }
        else:
            print(f"Skipping table from page {current_page_num} ({source_method}) due to no valid headers or data rows after final processing.")
            return None


    def detect_tables_from_pdf(self, pdf_file: Union[str, bytes], max_pages: Optional[int] = None, encoding : str = "utf-8") -> List[Dict]:
        # Reset the state and database at the beginning of each new detection call
        self._reset_state() # <<< ADDED THIS LINE
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_idx in range(min(len(pdf.pages), max_pages) if max_pages else len(pdf.pages)):
                page = pdf.pages[page_idx]
                current_page_num = page_idx + 1
                page_tables_found = []

                print(f"\n--- Processing Page {current_page_num} for tables ---")

                # Strategy 1: Try pdfplumber's native table extraction
                try:
                    print(f"Attempting pdfplumber native table extraction for page {current_page_num}...")
                    plumber_tables = page.extract_tables()
                    for idx, table_data in enumerate(plumber_tables):
                        if table_data and len(table_data) > 1 and any(cell.strip() if cell else '' for row in table_data[1:] for cell in row):
                            df_plumber = pd.DataFrame(table_data[1:], columns=table_data[0]) 
                            processed_table = self._process_extracted_dataframe(df_plumber, current_page_num, f"pdfplumber_table_{idx}")
                            if processed_table:
                                page_tables_found.append(processed_table)
                                print(f"  > Successfully extracted 1 table (pdfplumber native) from page {current_page_num}.")
                        else:
                            print(f"  > pdfplumber native table {idx} on page {current_page_num} yielded no valid data after initial check.")
                except Exception as e:
                    print(f"  > pdfplumber native table extraction failed for page {current_page_num}: {e}")

                # Strategy 2: If no tables from pdfplumber, try tabula-py (stream mode first for robustness)
                if not page_tables_found:
                    print(f"  > No tables from pdfplumber. Attempting tabula-py (stream mode) for page {current_page_num}...")
                    try:
                        tabula_dfs_stream = tabula.read_pdf(
                            pdf_file,
                            pages=current_page_num,
                            multiple_tables=True,
                            output_format="dataframe",
                            stream=True,
                            guess=True,
                            encoding='latin-1',
                            pandas_options={'header': None, 'on_bad_lines': 'warn'}
                        )
                        for idx, df_tabula_stream in enumerate(tabula_dfs_stream):
                            processed_table = self._process_extracted_dataframe(df_tabula_stream, current_page_num, f"tabula_stream_table_{idx}")
                            if processed_table:
                                page_tables_found.append(processed_table)
                                print(f"  > Successfully extracted 1 table (tabula stream) from page {current_page_num}.")
                    except tabula.errors.JavaTabulaError as e:
                        print(f"  > tabula-py (stream) failed for page {current_page_num} with Java error: {e}. Trying lattice mode...")
                    except Exception as e:
                        print(f"  > tabula-py (stream) failed for page {current_page_num}: {e}. Trying lattice mode...")

                # Strategy 3: If still no tables, try tabula-py (lattice mode) as a last automatic resort
                if not page_tables_found:
                    print(f"  > No tables from stream mode. Attempting tabula-py (lattice mode) for page {current_page_num}...")
                    try:
                        tabula_dfs_lattice = tabula.read_pdf(
                            pdf_file,
                            pages=current_page_num,
                            multiple_tables=True,
                            output_format="dataframe",
                            lattice=True,
                            guess=True,
                            encoding='latin-1',
                            pandas_options={'header': None, 'on_bad_lines': 'warn'}
                        )
                        for idx, df_tabula_lattice in enumerate(tabula_dfs_lattice):
                            processed_table = self._process_extracted_dataframe(df_tabula_lattice, current_page_num, f"tabula_lattice_table_{idx}")
                            if processed_table:
                                page_tables_found.append(processed_table)
                                print(f"  > Successfully extracted 1 table (tabula lattice) from page {current_page_num}.")
                    except tabula.errors.JavaTabulaError as e:
                        print(f"  > tabula-py (lattice) failed for page {current_page_num} with Java error: {e}.")
                    except Exception as e:
                        print(f"  > tabula-py (lattice) failed for page {current_page_num}: {e}.")

                # Add found tables to the main list and assign table_index
                if page_tables_found:
                    for table_dict in page_tables_found:
                        table_dict['table_index'] = len(self.tables)
                        self.tables.append(table_dict)
                else:
                    print(f"  > No tables successfully extracted from page {current_page_num} after all automated attempts.")

        self.infer_schemas()
        return self.tables

    def detect_key_value_pairs(self, pdf_file: Union[str, bytes], max_pages: Optional[int] = None) -> List[Dict]:
        self.kv_pairs = []
        pattern = re.compile(r'^(.*?)\s*[:=\-\u2013\u2014]\s*(.*?)(?=\n|$)', re.MULTILINE)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_idx, page in enumerate(pdf.pages[:max_pages] if max_pages else pdf.pages):
                    try:
                        text = page.extract_text() or ""
                        
                        if page.images:
                            for i, img_data in enumerate(page.images):
                                try:
                                    if 'stream' in img_data and hasattr(img_data['stream'], 'rawdata'):
                                        img = Image.open(io.BytesIO(img_data['stream'].rawdata))
                                        extracted_text = pytesseract.image_to_string(img)
                                        if extracted_text:
                                            text += extracted_text + "\n"
                                except Exception as e:
                                    print(f"Error extracting text from image {i} on page {page_idx+1}: {str(e)}. This might be due to unsupported image formats or corrupted image data.")

                        if not text.strip():
                            print(f"No meaningful text extracted from page {page_idx + 1} for KV pairs.")
                            continue

                        matches = pattern.findall(text)
                        self.kv_pairs.extend([
                            {"Key": k.strip(), "Value": v.strip(), "source_page": page_idx+1}
                            for k, v in matches if k.strip() and v.strip()
                        ])

                    except Exception as err:
                        print(f"Error during key-value extraction on page {page_idx + 1}: {err}")

            return self.kv_pairs
        except Exception as e:
            raise Exception(f"Key-value extraction failed: {str(e)}")

    def detect_critical_values(self, pdf_file: Union[str, bytes]) -> Dict[str, Any]:
        self.critical_data = {}
        patterns = {
            "total": r"Total\s*[:=\-]\s*(\$?\d[\d,]*(?:\.\d{2})?)",
            "date": r"(?:Date|As of)\s*[:=\-]\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "invoice": r"(?:Invoice|Bill)\s*#?\s*[:=\-]\s*([A-Z0-9\-]+)",
            "amount_due": r"(?:Amount Due|Balance)\s*[:=\-]\s*(\$?\d[\d,]*(?:\.\d{2})?)"
        }
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                        if page.images:
                            for i, img_data in enumerate(page.images):
                                try:
                                    if 'stream' in img_data and hasattr(img_data['stream'], 'rawdata'):
                                        img = Image.open(io.BytesIO(img_data['stream'].rawdata))
                                        extracted_text = pytesseract.image_to_string(img)
                                        if extracted_text:
                                            text += extracted_text + "\n"
                                except Exception as e:
                                    print(f"Error extracting text from image {i} on page {page_idx+1}: {str(e)}. This might be due to unsupported image formats or corrupted image data.")
                        if not text.strip():
                            print(f"No meaningful text extracted from page {page_idx + 1} for critical values.")
                            continue

                    except Exception as err:
                        print(f"Error during critical value text extraction on page {page_idx + 1}: {err}")
                        continue

                    for key, pattern in patterns.items():
                        if key not in self.critical_data:
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                self.critical_data[key] = match.group(1)
            return self.critical_data
        except Exception as e:
            raise Exception(f"Critical value extraction failed: {str(e)}")

    def get_table_preview(self, table_idx: int = 0) -> pd.DataFrame:
        if not self.schemas or table_idx >= len(self.schemas):
            print(f"No schema found for table index {table_idx} or index out of range.")
            return pd.DataFrame()
        
        try:
            table_name = self.schemas[table_idx]["table_name"]
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 5')
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
            if df.empty:
                print(f"Warning: Preview for table {table_idx} ({table_name}) is empty from SQLite. This table might have no data rows.")
            return df
        except Exception as e:
            print(f"Error fetching preview from SQLite for table {table_idx}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()


    def save_table_to_csv(self, table_idx: int, filename: str = None) -> str:
        if table_idx >= len(self.schemas):
            raise IndexError(f"Table index {table_idx} out of range for saving.")
            
        table_name = self.schemas[table_idx]["table_name"]
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', self.conn)
        except Exception as e:
            print(f"Error reading from SQLite for table {table_name} for saving: {e}.")
            raise ValueError(f"Failed to read data for table '{table_name}' from database.")

        if df.empty:
            print(f"Warning: Attempted to save an empty DataFrame for table {table_idx} ({table_name}).")
            raise ValueError("No data available to save for this table.")
            
        filename = filename or f"table_{table_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.storage_dir, self._sanitize_filename(filename))
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Table {table_idx} ({table_name}) saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving table {table_idx} ({table_name}) to CSV: {e}")
            raise Exception(f"Failed to save CSV for table '{table_name}'.")

    def save_all_tables(self) -> List[str]:
        saved_files = []
        if not self.schemas:
            print("No tables (schemas) extracted to save.")
            return saved_files
            
        for i in range(len(self.schemas)):
            try:
                filepath = self.save_table_to_csv(i)
                saved_files.append(filepath)
            except ValueError as ve:
                print(f"Skipping save for table {i}: {ve}")
            except Exception as e:
                print(f"Error saving table {i}: {str(e)}")
        
        if not saved_files:
            print("No tables were successfully saved.")
        return saved_files


    def infer_schemas(self) -> List[Dict]:
        self.schemas = [] # Re-initialize self.schemas here as well
        
        # Explicitly close and remove the SQLite database file
        try:
            if hasattr(self._local, "conn") and self._local.conn:
                self._local.conn.close()
                del self._local.conn
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                print(f"Cleaned up old SQLite database: {self.db_path}")
        except Exception as e:
            print(f"Warning: Could not fully clean up old SQLite database: {e}")
        
        try:
            for i, table_data in enumerate(self.tables):
                schema = {
                    'table_name': f'extracted_table_{i+1}',
                    'columns': [],
                    'source_page': table_data.get('source_page', 0)
                }
                
                temp_df = pd.DataFrame(table_data['rows'], columns=table_data['columns'])

                for col_name in table_data['columns']:
                    col_type = 'TEXT'
                    if col_name in temp_df.columns:
                        col_series = temp_df[col_name].dropna()
                        if not col_series.empty:
                            try:
                                numeric_series = pd.to_numeric(col_series, errors='coerce').dropna()
                                if not numeric_series.empty:
                                    if pd.api.types.is_integer_dtype(numeric_series):
                                        col_type = 'INTEGER'
                                    elif pd.api.types.is_float_dtype(numeric_series):
                                        col_type = 'REAL'
                            except Exception:
                                pass

                    schema['columns'].append({
                        'name': col_name,
                        'type': col_type,
                        'sample_values': temp_df[col_name].head(3).tolist() if col_name in temp_df.columns else []
                    })
                
                self.schemas.append(schema)
                self._create_sqlite_table(schema, table_data['rows'])

            if self.kv_pairs:
                schema_kv = {
                    'table_name': 'key_value_pairs',
                    'columns': [
                        {'name': 'Key', 'type': 'TEXT'},
                        {'name': 'Value', 'type': 'TEXT'},
                        {'name': 'source_page', 'type': 'INTEGER'}
                    ]
                }
                self.schemas.append(schema_kv)
                kv_rows = [[p['Key'], p['Value'], p.get('source_page', None)] for p in self.kv_pairs]
                self._create_sqlite_table(schema_kv, kv_rows)

            if self.critical_data:
                schema_crit = {
                    'table_name': 'critical_data',
                    'columns': [
                        {'name': 'Key', 'type': 'TEXT'},
                        {'name': 'Value', 'type': 'TEXT'}
                    ]
                }
                self.schemas.append(schema_crit)
                crit_rows = [[k, v] for k, v in self.critical_data.items()]
                self._create_sqlite_table(schema_crit, crit_rows)

            print(f"Inferred schemas for {len(self.schemas)} tables/data types.")
            return self.schemas
        except Exception as e:
            raise Exception(f"Schema inference failed: {str(e)}")

    def _create_sqlite_table(self, schema: Dict, rows: List[List]) -> None:
        cursor = self.conn.cursor()
        try:
            # DROP TABLE IF EXISTS is already here, but if the connection is not new,
            # or if previous transaction failed, it might cause issues.
            # Explicitly drop just to be safe, though a clean DB on infer_schemas is better.
            cursor.execute(f'DROP TABLE IF EXISTS "{schema["table_name"]}"')

            columns_def = ', '.join(f'"{col["name"]}" {col["type"]}' for col in schema['columns'])
            cursor.execute(f'CREATE TABLE "{schema["table_name"]}" ({columns_def})')

            num_cols = len(schema['columns'])

            if rows: # Only attempt insert if there are rows
                for row in rows:
                    padded_row = row + [None] * (num_cols - len(row))
                    truncated_row = padded_row[:num_cols]
                    final_row_for_db = [str(x) if x is not None else None for x in truncated_row]

                    placeholders = ', '.join(['?'] * num_cols)
                    cursor.execute(
                        f'INSERT INTO "{schema["table_name"]}" VALUES ({placeholders})',
                        final_row_for_db
                    )
                self.conn.commit()
                print(f"Successfully created and populated SQLite table: '{schema['table_name']}' with {len(rows)} rows.")
            else:
                self.conn.commit() # Commit create table even if no rows
                print(f"Successfully created empty SQLite table: '{schema['table_name']}'.")

        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"SQLite error creating/populating table '{schema['table_name']}': {e}")
            raise Exception(f"Failed to create table '{schema['table_name']}': {str(e)}")

    def execute_sql_query(self, query: str) -> Union[pd.DataFrame, str]:
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                return pd.DataFrame(
                    cursor.fetchall(),
                    columns=[desc[0] for desc in cursor.description]
                )
            self.conn.commit()
            return f"Query executed. Rows affected: {cursor.rowcount}"
        except sqlite3.Error as e:
            return f"Query failed: {str(e)}"

    def export_schema(self) -> str:
        concise_schemas = []
        for schema in self.schemas:
            concise_columns = []
            for col in schema['columns']:
                concise_columns.append({
                    'name': col['name'],
                    'type': col['type']
                })
            concise_schemas.append({
                'table_name': schema['table_name'],
                'columns': concise_columns
            })

        summary = {
            'num_tables_extracted': len(self.tables),
            'num_key_value_pairs_extracted': len(self.kv_pairs),
            'num_critical_values_extracted': len(self.critical_data)
        }

        return json.dumps({
            'summary': summary,
            'tables': concise_schemas,
            'note': 'Sample values are omitted for brevity to fit model context. Assume standard SQL types.'
        }, indent=2)

    def cleanup(self):
        if hasattr(self._local, "conn"):
            try:
                self._local.conn.close()
                del self._local.conn
            except Exception as e:
                print(f"Error during SQLite connection cleanup: {e}")
        
        try:
            if jpype.isJVMStarted():
                jpype.shutdownJVM()
                print("JVM shutdown.")
        except Exception as e:
            print(f"Error during JVM shutdown: {e}")

    def __del__(self):
        self.cleanup()
