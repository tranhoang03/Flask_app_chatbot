import requests
from flask import Flask, request, jsonify
from models.rag_system import OptimizedRAGSystem
from config import Config

app = Flask(__name__)

# Initialize RAG system
config = Config()
rag_system = OptimizedRAGSystem(config)

# URL của API OCR đang chạy trên localhost
# API_URL = "https://1332-34-143-143-231.ngrok-free.app/ocr"


# def send_image(image_url):
#     payload = {"image_url": image_url}
#     response = requests.post(API_URL, json=payload)

    
#     if response.status_code == 200:

#         print("Response:", response.json()["response_message"])
#     else:
#         print("Error:", response.status_code, response.text)

# image_url = input("Nhập URL hình ảnh: ")
# send_image(image_url)

# import requests 
# class get_info: 
#     def send_image(file_path,API_URL):
#         files = {'image': open(file_path, 'rb')}
#         response = requests.post(API_URL, files=files)
#         if response.status_code == 200:
#             print("Response:", response.json()["response_message"])
#         else:
#             print("Error:", response.status_code, response.text)

# API_URL = "https://278b-35-185-225-64.ngrok-free.app/ocr" 
# file_path = input("Nhập đường dẫn ảnh trên máy: ")
# get_info.send_image(file_path,API_URL)

import requests

class get_info:
    @staticmethod
    def send_image(file_path, API_URL):
        try:
            # Nếu là ảnh URL (gửi dưới dạng JSON)
            if file_path.startswith('http://') or file_path.startswith('https://'):
                response = requests.post(API_URL, json={"image_url": file_path})
            else:
                # Nếu là ảnh local (upload file)
                with open(file_path, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(API_URL, files=files)

            # Kiểm tra phản hồi
            if response.status_code == 200:
                response_json = response.json()
                return response_json["response_message"]
            else:
                print("Lỗi từ OCR API:", response.status_code, response.text)
                return None

        except Exception as e:
            print("Lỗi khi gửi ảnh:", e)
            return None
