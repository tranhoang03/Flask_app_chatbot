import json
import sqlite3
from typing import List, Dict, Any, Tuple
import base64
import os
import re

def load_table_data(db_path: str) -> List[Dict[str, Any]]:
    """Load data from all tables in the database and format for vector store"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        documents = []
        
        column_name_mapping = {
            # Bảng Categories
            "Id": "id danh mục",
            "Name": "tên danh mục", 
            "Description": "mô tả danh mục",

            # Bảng Product
            "Categories_id": "id danh mục",
            # "Id": "id sản phẩm",       
            # "Name": "tên sản phẩm",       
            "Product_Prep": "thành phần sản phẩm",
            "Calories": "calo",
            "Dietary_Fibre_g": "chất xơ",
            "Sugars_g": "đường",
            "Protein_g": "protein",
            "Vitamin_A": "vitamin A",
            "Vitamin_C": "vitamin C",
            "Caffeine_mg": "caffeine",
            "Price": "đơn giá",
            "Sales_rank": "hạng bán chạy",
            "Descriptions": "mô tả sản phẩm",
            "Link_Image": "link ảnh",

            # Bảng Store
            # "Id": "id cửa hàng",          
            # "Name": "tên cửa hàng",        
            "Address": "địa chỉ",
            "Phone": "số điện thoại",
            "Open_Close": "giờ mở cửa đóng cửa",

            # Bảng Orders
            # "Id": "id đơn hàng",        
            "Customer_id": "id khách hàng",
            "Store_id": "id cửa hàng",
            "Order_date": "ngày đặt hàng",

            # Bảng Order_detail
            "Order_id": "id đơn hàng",
            "Product_id": "id sản phẩm",
            "Quantity": "số lượng",
            "Price": "đơn giá",
            "Rate": "đánh giá", # Hoặc "đánh giá"

            # Bảng Customer_preferences
          
            "Preferred_categories": "danh mục ưa thích",
            "Max_price": "giá tối đa",

            # Bảng customers
            "id": "id khách hàng", 
            "name": "tên khách hàng",
            "sex": "giới tính",
            "age": "tuổi",
            "location": "địa chỉ", 
            "picture": "ảnh",            
            "embedding": "embedding"     
        }

        print("\n=== Loading Data for Vector Store ===")
        for table_tuple in tables:
            table_name = table_tuple[0]
            print(f"Processing table: {table_name}")
            

            if table_name == 'sqlite_sequence':
                continue

            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            
            print(f"  - Found {len(rows)} rows")
            
            # Convert each row to a document
            for row_idx, row in enumerate(rows):
                # Create a dictionary of column names and values
                row_dict = {}
                for col_name, val in zip(column_names, row):
                    if (table_name == "customers" and col_name in ["embedding", "picture"]) or \
                       (table_name == "Product" and col_name == "Link_Image"):
                        continue
                    row_dict[col_name] = val

                content_parts = []
                for k, v in row_dict.items():
                    display_name = column_name_mapping.get(k, k) 
                    value_str = str(v) if v is not None else "không có"
                    content_parts.append(f"{display_name}: {value_str}")

                content = f"Bảng {table_name}: " + ", ".join(content_parts)
                
                metadata = {
                    "table": table_name,
                     "columns": list(row_dict.keys()), 
                    "data": row_dict, 
                    "original_row_index": row_idx 
                }
                
                documents.append({
                    "content": content,
                    "metadata": metadata
                })
        
        print("\n=== Summary ===")
        print(f"Total documents created: {len(documents)}")
        print("="*50)
        
        conn.close()
        return documents
        
    except Exception as e:
        print(f"Error loading table data: {e}")
        return []



def execute_sql_query(db_path: str, query: str, timeout: int = 30) -> List[Dict[str, Any]]:
    """Execute SQL query and return results"""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        
        cursor.execute(query)
        
  
        columns = [description[0] for description in cursor.description]

        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            result_dict = dict(zip(columns, row))
            results.append(result_dict)
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return []

def format_sql_results(results: List[Dict[str, Any]]) -> str:
    """Format SQL results into a readable string"""
    if not results:
        return "Không tìm thấy kết quả"
    
    formatted_results = []
    for result in results:
   
        result_str = ", ".join([
            f"{k}: {v}" for k, v in result.items()
        ])
        formatted_results.append(result_str)
    
    return "\n".join(formatted_results)


def validate_sql_query(query: str) -> bool:
    """Validate SQL query for safety and basic correctness."""
    
    def log_invalid(reason: str) -> bool:
        print(f"Validation failed: {reason}")
        return False

    try:
        if not query or not query.strip():
            return log_invalid("Empty query")

        query_norm = query.strip()
        query_upper = query_norm.upper()

        # Cho phép dấu ; nếu nó ở cuối câu và chỉ xuất hiện một lần
        semicolon_count = query_norm.count(';')
        if semicolon_count > 1:
            return log_invalid("Multiple semicolons detected")
        if semicolon_count == 1 and not query_norm.endswith(';'):
            return log_invalid("Semicolon not at end of query")

        if query_norm.endswith(';'):
            query_norm = query_norm[:-1].rstrip()
            query_upper = query_norm.upper()

        if not query_upper.startswith("SELECT"):
            return log_invalid(f"Not a SELECT statement: {query_norm}")

    
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        if any(re.search(rf"\b{kw}\b", query_upper) for kw in dangerous_keywords):
            return log_invalid(f"Dangerous keyword found: {query_norm}")

        if '--' in query_norm:
            return log_invalid("Potential comment injection ('--')")


        if "FROM" not in query_upper:
            return log_invalid("Missing FROM clause")


        if query_norm.count('(') != query_norm.count(')'):
            return log_invalid("Unbalanced parentheses")

        # Basic SELECT ... FROM syntax
        if not re.match(r"(?i)^SELECT\s+.+\s+FROM\s+.+", query_norm):
            return log_invalid("Malformed SELECT syntax")

        print(f"Query validation passed: {query_norm}")
        return True

    except Exception as e:
        print(f"Error during SQL query validation: {str(e)}")
        return False

def get_purchase_history(user_id: int) -> list:
    """Fetches the last 5 purchase history items for a given user ID."""
    try:
        # Assuming Database.db is in the root directory relative to where app is run
        db_path = os.path.join(os.path.dirname(__file__), '..', 'Database.db')
        if not os.path.exists(db_path):
             # If not in root, maybe it's inside the flask folder?
             db_path = os.path.join(os.path.dirname(__file__), 'Database.db')
             if not os.path.exists(db_path):
                 print(f"Database file not found at expected locations.")
                 return []

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """
            SELECT o.Order_date, p.Name, od.Quantity, (p.Price * od.Quantity) AS Price, od.Rate
            FROM Orders o
            JOIN Order_detail od ON o.Id = od.Order_id
            JOIN Product p ON od.Product_id = p.Id
            WHERE o.Customer_id = ?
            ORDER BY o.Order_date DESC
            LIMIT 5
        """

        cursor.execute(query, (user_id,))
        results = cursor.fetchall()
        conn.close()
        history = [
            {"date": row[0], "product": row[1], "quantity": row[2], "price": row[3], "rate": row[4]}
            for row in results
        ]
        return history
    except Exception as e:
        print(f"Error getting purchase history for user {user_id}: {e}")
        return []
