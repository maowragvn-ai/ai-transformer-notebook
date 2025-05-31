from typing import Any, Dict, List
import underthesea
import pandas as pd
import re
import numpy as np
import json
import os
from docx import Document
from .stop_word import load_vietnamese_stopwords
from underthesea import word_tokenize, sent_tokenize
def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
def read_docx_file(path: str) -> str:
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

class DataPreprocessor:
    # Pattern để nhận diện điều luật
    article_patterns = [
        r'Điều\s+(\d+)\.?\s+(.*?)\n(.*?)(?=\nĐiều\s+\d+\.|\Z)',
        r'Điều\s+(\d+[a-z]?)\.?\s*(.*?)(?=Điều\s+\d+|$)',
        r'Article\s+(\d+)\.?\s*(.*?)(?=Article\s+\d+|$)'
    ]
    def __init__(self, stopwords: list = None):
        self.stopwords = stopwords or load_vietnamese_stopwords()
        self.output_path = os.path.join("data", 'processed')
        os.makedirs(self.output_path, exist_ok=True)
    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s\.,;:\-\(\)\[\]\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    def normalize_currency(self, text: str) -> str:
        """Chuẩn hóa định dạng tiền vnd"""
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*triệu', lambda m: f"{float(m.group(1)) * 1000000:.0f} VNĐ"),
            (r'(\d+(?:\.\d+)?)\s*tỷ', lambda m: f"{float(m.group(1)) * 1000000000:.0f} VNĐ"),
            (r'(\d+(?:\.\d+)?)\s*m(?:\s|$)', lambda m: f"{float(m.group(1)) * 1000000:.0f} VNĐ"),
            (r'(\d+(?:\.\d+)?)\s*k(?:\s|$)', lambda m: f"{float(m.group(1)) * 1000:.0f} VNĐ"),
            (r'(\d+)\.(\d{3})\.(\d{3})', r'\1.\2.\3 VNĐ')
        ]
        for pattern, replacement in patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    def remove_stop_words(self, text: str) -> str:
        """Loại bỏ từ dừng (tùy chọn cho phân tích)"""
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        return ' '.join(filtered_tokens)
    def tokenize_vietnamese(self, text: str) -> Dict[str, List[str]]:
        return {
            'words': word_tokenize(text),
            'sentences': sent_tokenize(text)
        }
    def extract_articles(self, text: str, remove_stopwords: bool = False) -> Dict[str, Dict[str, Any]]:
        articles = {}

        pattern = re.compile(
            self.article_patterns[0],
            re.DOTALL | re.IGNORECASE
        )

        matches = pattern.finditer(text)

        for match in matches:
            article_number = match.group(1)
            title = self.clean_text(match.group(2))
            content = self.clean_text(match.group(3))
            content = self.normalize_currency(content)
            if remove_stopwords:
                content = self.remove_stop_words(content)
            articles[f"dieu_{article_number}"] = {
                'title': title,
                'text': content
            }

        return articles
    
    def process_document(self, file_path: str, remove_stopwords: bool = False) -> Dict:
        """Xử lý toàn bộ tài liệu"""
        if not file_path:
            print("File path is empty")
            raise ValueError("File path cannot be empty")
        file_name = os.path.basename(file_path)
        # Đọc file
        raw_text = read_docx_file(file_path)
        
        if not raw_text:
            print(f"Không thể đọc file: {file_name}")
            return {"error": "Không thể đọc file"}
        print(f"Đã đọc file: {file_name}")
        
        # Trích xuất các điều luật
        articles = self.extract_articles(raw_text, remove_stopwords)
        print(f"Đã trích xuất {len(articles)} điều luật")
        file_to_save = os.path.join(self.output_path, f'{file_name.split(".")[0]}.json')
        print(f"Đã lưu file: {file_to_save}")
        self.save_to_json(articles, file_to_save)
        return articles
    
    def save_to_json(self, processed_data: Dict[str, Any], output_path: str):
        """Lưu dữ liệu đã xử lý ra file JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    # data_preprocessor= DataPreprocessor()
    # data_preprocessor.process_document('data/input/41_2024_QH15_557190.docx')
    #print(len(data))
    list_files = os.listdir('data/input')
    data_preprocessor = DataPreprocessor()
    for file_name in list_files:
        if file_name.endswith('.docx'):
            file_path = os.path.join('data/input', file_name)
            data_preprocessor.process_document(file_path)
