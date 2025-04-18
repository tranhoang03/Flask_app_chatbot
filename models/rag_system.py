from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os

import sqlite3
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings 

from transformers import AutoModel, AutoTokenizer
import torch
from config import Config
from utils import (
    load_table_data,
    execute_sql_query,
    format_sql_results,
    validate_sql_query,
    get_purchase_history
)
from .chat_history import ChatHistory
from .prompts import PromptManager
class PhoBERTEmbeddings(Embeddings):
    def __init__(self, model_name: str = "vinai/phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Set a default max length
        self.max_length = 256 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Add max_length to tokenizer call
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Ensure output is correctly handled, even if sequence length is 1 after mean
            mean_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy() 
            embeddings.append(mean_emb.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Add max_length to tokenizer call
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Ensure output is correctly handled
        mean_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        return mean_emb.tolist()


class OptimizedRAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.chat_history = ChatHistory()
        self._initialize_components()
    
    def _initialize_components(self):
        """Khởi tạo các thành phần chính"""
        self.embeddings = PhoBERTEmbeddings()

        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            google_api_key=self.config.google_api_key
        )
        
        self.vector_store = self._initialize_vector_store()
        self.description_vector_store = self._initialize_description_vector_store()

    def _initialize_vector_store(self) -> FAISS:
        """Load vector store or create new if not found"""
        if os.path.exists(self.config.vector_store_path):
            try:
                return FAISS.load_local(
                    self.config.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return self._create_new_vector_store()
        return self._create_new_vector_store()

    def _create_new_vector_store(self) -> FAISS:
        """Create a new FAISS vector store"""
        try:
            os.makedirs(self.config.vector_store_path, exist_ok=True)
            documents = load_table_data(self.config.db_path)
            if not documents:
                print("No documents loaded from database")
                return None
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            print(f"Creating vector store with {len(texts)} documents")
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            vector_store.save_local(self.config.vector_store_path)
            print("Vector store created and saved successfully")
            return vector_store

        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def _initialize_description_vector_store(self) -> FAISS:
        """Initialize or create description vector store"""
        if os.path.exists(self.config.description_vector_store_path):
            try:
                return FAISS.load_local(
                    self.config.description_vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading description vector store: {e}")
                return self._create_description_vector_store()
        return self._create_description_vector_store()

    def _create_description_vector_store(self) -> FAISS:
        """Create FAISS vector store for product descriptions"""
        try:
            os.makedirs(self.config.description_vector_store_path, exist_ok=True)
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT Name_Product, Descriptions FROM Product;")
            products = cursor.fetchall()
            conn.close()

            if not products:
                print("No product descriptions found.")
                return None

            descriptions = [
                f"{product[1]}" for product in products
            ]
            metadatas = [
                {"name": product[0], "description": product[1]} for product in products
            ]

            vector_store = FAISS.from_texts(
                texts=descriptions,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            vector_store.save_local(self.config.description_vector_store_path)
            print("Description vector store created and saved successfully.")
            return vector_store

        except Exception as e:
            print(f"Error creating description vector store: {e}")
            return None
   
    def _needs_calculation(self, query: str, user_key: str) -> bool:
        """Check if query requires calculation using LLM"""
        prompt = f"""
        Bạn là một chuyên gia phân loại yêu cầu của người dùng để chọn phương pháp xử lý tối ưu nhất. Hãy phân tích kỹ câu hỏi sau và quyết định xem nên dùng phương pháp nào để trả lời:

        Câu hỏi: "{query}"
        Lịch sử trò chuyện: {self.chat_history.get_recent_history(user_key)}
        Hệ thống có 2 phương pháp chính để tìm câu trả lời:

        1.  **Database (SQL):** Dùng khi câu hỏi yêu cầu truy xuất, tính toán, hoặc tổng hợp **dữ liệu có cấu trúc** từ cơ sở dữ liệu.
            *   **Mục đích:** Lấy dữ liệu chính xác, thực hiện các phép tính (tổng, trung bình, đếm), lọc, sắp xếp, so sánh, thống kê.
            *   **Ví dụ câu hỏi phù hợp:**
                *   "Tính tổng doanh thu tháng 5."
                *   "Liệt kê 3 sản phẩm bán chạy nhất tuần trước."
                *   "Có bao nhiêu đơn hàng được giao trong ngày hôm qua?"
                *   "Cho tôi danh sách các loại trà sữa giá dưới 50 nghìn đồng."
                *   "So sánh doanh số giữa chi nhánh A và B."
                *   "Mặt hàng nào được bán nhiều nhất?"
                .....

        2.  **Vector Store:** Dùng khi câu hỏi mang tính **hỏi đáp thông tin chung, mô tả, giải thích, hoặc tìm kiếm theo ngữ nghĩa** mà không yêu cầu tính toán phức tạp trên dữ liệu có cấu trúc.
            *   **Mục đích:** Tìm kiếm thông tin liên quan dựa trên ý nghĩa, trả lời các câu hỏi về đặc điểm sản phẩm, quy trình, thông tin cửa hàng, hoặc các chủ đề chung.
            *   **Ví dụ câu hỏi phù hợp:**
                *   "Trà sữa trân châu đường đen có vị như thế nào?"
                *   "Cửa hàng mình có chỗ để xe máy không?"
                *   "Giải thích cách pha cà phê phin."
                *   "Cho tôi gợi ý đồ uống giải nhiệt mùa hè."
                *   "Thành phần dinh dưỡng của món ABC là gì?"
                *   "Giờ mở cửa của chi nhánh X?" (Nếu giờ mở cửa là thông tin cố định, không cần truy vấn động)
                ....

        **Các trường hợp KHÔNG nên dùng SQL (nên dùng Vector Store - trả về "false"):**
        *   Câu chào hỏi, cảm ơn đơn thuần ("Chào bạn", "Cảm ơn")
        *   Câu hỏi thể hiện cảm xúc ("Tôi thấy hơi mệt")
        *   Câu hỏi rất chung chung, không yêu cầu dữ liệu cụ thể ("Bạn có thể làm gì?", "Kể chuyện cười đi")
        *   Câu hỏi về cách sử dụng hệ thống hoặc về chính bạn ("Bạn là ai?")

        **Yêu cầu:**
        1.  Phân tích kỹ câu hỏi của người dùng.
        2.  Quyết định phương pháp **phù hợp và hiệu quả nhất** (SQL hay Vector Store).
        3.  **Chỉ trả về "true"** nếu bạn quyết định dùng **Database (SQL)**.
        4.  **Chỉ trả về "false"** nếu bạn quyết định dùng **Vector Store**.
        5.  **KHÔNG** giải thích gì thêm, chỉ trả về "true" hoặc "false".
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Extract only the content from the response
            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            else:
                result = str(response).strip().lower()
            
            return result == "true"
            
        except Exception as e:
            print(f"Error in _needs_calculation: {e}")
            # Fallback to simple keyword check if LLM fails
            calculation_keywords = [
                "tính", "tổng", "trung bình", "số lượng", "count", "sum", "average",
                "nhiều nhất", "ít nhất", "max", "min", "so sánh", "thống kê",
                "danh sách", "liệt kê", "hiển thị", "show", "list", "display"
            ]
            return any(keyword in query.lower() for keyword in calculation_keywords)
    
    def _get_database_schema(self) -> str:
        """Get database schema information"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_info = []
            for table in tables:
                table_name = table[0]
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = cursor.fetchall()
                
                # Get indexes (including primary keys)
                cursor.execute(f"PRAGMA index_list({table_name})")
                indexes = cursor.fetchall()
                
                # Format column information
                column_info = []
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = col[5] == 1  # Check if column is primary key
                    pk_info = " (PRIMARY KEY)" if is_pk else ""
                    column_info.append(f"{col_name} ({col_type}){pk_info}")
                
                # Format foreign key information
                fk_info = []
                for fk in foreign_keys:
                    ref_table = fk[2]  # Referenced table
                    from_col = fk[3]   # Column in this table
                    to_col = fk[4]     # Column in referenced table
                    fk_info.append(f"FOREIGN KEY ({from_col}) REFERENCES {ref_table}({to_col})")
                
                # Format index information
                index_info = []
                for idx in indexes:
                    idx_name = idx[1]
                    is_unique = idx[2] == 1
                    if not idx_name.startswith('sqlite_autoindex'):  # Skip auto-generated indexes
                        index_info.append(f"{'UNIQUE ' if is_unique else ''}INDEX {idx_name}")
                
                # Combine all information
                table_info = [f"Bảng {table_name}:"]
                table_info.extend(column_info)
                if fk_info:
                    table_info.append("\nKhóa ngoại:")
                    table_info.extend(fk_info)
                if index_info:
                    table_info.append("\nChỉ mục:")
                    table_info.extend(index_info)
                
                schema_info.append("\n".join(table_info))
            
            conn.close()
            return "\n\n".join(schema_info)
            
        except Exception as e:
            print(f"Error getting database schema: {e}")
            return ""

    
    def _answer_with_vector(self, user_key: str, query: str, user_info: dict, purchase_history: list, is_image_upload: bool = False) -> str:
        """Answer query using vector search, with a special prompt for image uploads"""
        try:
            # Chọn vector store phù hợp
            if is_image_upload:
                if not self.description_vector_store:
                    return "Không thể tìm kiếm vì vector store mô tả chưa sẵn sàng."
                vector_store = self.description_vector_store
            else:
                vector_store = self.vector_store

            # Truy xuất tài liệu liên quan
            docs = vector_store.similarity_search(
                query,
                k=self.config.top_k_results
            )
            if is_image_upload:
                context = [
                    f"Tên: {doc.metadata['name']}, Mô tả: {doc.page_content}"
                    for doc in docs
                ]
            else:
                context = [doc.page_content for doc in docs]

            # Lấy lịch sử chat gần đây
            recent_history = self.chat_history.get_recent_history(user_key)

            # Tạo prompt tùy theo loại truy vấn
            if is_image_upload:
                prompt = PromptManager.get_image_upload_prompt(context, query, recent_history, user_info)
            else:
                prompt = PromptManager.get_vector_prompt(context, query, recent_history, user_info, purchase_history)

            print(prompt)  # Debug
            response = self.llm.invoke(prompt)

            return getattr(response, "content", str(response)).strip()

        except Exception as e:
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"

    
    def _answer_with_sql(self, user_key: str, query: str, user_info: dict, purchase_history: list) -> str:
        """Answer query using SQL"""
        try:
            # 1. Tạo prompt để sinh câu lệnh SQL từ LLM
            sql_prompt = PromptManager.get_sql_generation_prompt(query, self._get_database_schema())

            # 2. Gọi LLM sinh câu lệnh SQL
            sql_query_response = self.llm.invoke(sql_prompt)

            # 3. Lấy đúng phần nội dung SQL (không lấy cả metadata)
            sql_query_string = sql_query_response.content.strip() if hasattr(sql_query_response, 'content') else str(sql_query_response).strip()
            print("Generated SQL query:", sql_query_string)

            if not validate_sql_query(sql_query_string):
                return "Xin lỗi, tôi không thể thực hiện truy vấn này vì lý do an toàn hoặc truy vấn không hợp lệ."

            # 5. Thực thi câu lệnh SQL
            results = execute_sql_query(
                self.config.db_path,
                sql_query_string,
                self.config.db_timeout
            )

      
            formatted_results = format_sql_results(results)    
            recent_history = self.chat_history.get_recent_history(user_key)

            response_prompt = PromptManager.get_sql_response_prompt(
                query=query,
                results=formatted_results,
                history=recent_history,
                user_info=user_info,
                purchase_history=purchase_history
            )
            print(response_prompt)
            final_response = self.llm.invoke(response_prompt)
            return final_response.content.strip() if hasattr(final_response, 'content') else str(final_response).strip()

        except Exception as e:
            print(f"Error during SQL processing: {e}")
            return f"Lỗi khi xử lý câu hỏi liên quan đến SQL: {str(e)}"

    
    def answer_query(self, user_key: str, query: str) -> str:
        """Process query and return answer"""
        try:

            needs_sql = self._needs_calculation(query, user_key)
            print(f"LLM decision: {'1' if needs_sql else '0'}")  
            
            # Fetch user information and purchase history
            user_info = self._get_user_info(user_key)
            purchase_history = get_purchase_history(user_key)
            
            if needs_sql:
                response = self._answer_with_sql(user_key, query, user_info, purchase_history)
            else:
                response = self._answer_with_vector(user_key, query, user_info, purchase_history)

            self.chat_history.add_chat(user_key, query, response)
            
            return response
                
        except Exception as e:
            error_msg = f"Lỗi hệ thống: {str(e)}"
            self.chat_history.add_chat(user_key, error_msg)
            return error_msg

    def _get_user_info(self, user_key: str) -> dict:
        """Fetch user information based on user key"""
        if user_key == "anonymous":
            return None

        try:

            db_path = os.path.join(os.path.dirname(__file__), '..', 'Database.db')
            if not os.path.exists(db_path):
                db_path = os.path.join(os.path.dirname(__file__), 'Database.db')
                if not os.path.exists(db_path):
                    print(f"Database file not found at expected locations.")
                    return None

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            query = "SELECT id, name, sex FROM Customers WHERE id = ?"
            cursor.execute(query, (user_key,))
            result = cursor.fetchone()
            conn.close()

            if result:
                return {"id": result[0], "name": result[1], "sex": result[2]}
            else:
                print(f"User with ID {user_key} not found.")
                return None
        except Exception as e:
            print(f"Error fetching user info for user {user_key}: {e}")
            return None

    def clear_chat_history(self, user_key: str):
        """Clears the chat history for a specific user key."""
        try:
            self.chat_history.clear_history(user_key)
        except Exception as e:
            print(f"Error clearing chat history for {user_key}: {e}")

    def _initialize_description_vector_store(self) -> FAISS:
        """Initialize description vector store if exists, otherwise create it"""
        if os.path.exists(self.config.description_vector_store_path):
            try:
                return FAISS.load_local(
                    self.config.description_vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading description vector store: {e}")
        
        return self._create_description_vector_store()
