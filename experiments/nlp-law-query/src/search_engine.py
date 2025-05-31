import numpy as np
from typing import List, Dict, Tuple, Any
import os
from sklearn.metrics.pairwise import cosine_similarity
from .stop_word import load_vietnamese_stopwords
from underthesea import word_tokenize
import re
from .embed import Embed
import torch

class SearchEngine:
    def __init__(self, model_name: str = "vinai/phobert-base-v2", batch_size: int = 32):
        self.stopwords = load_vietnamese_stopwords()
        self.embed = Embed(model_name, batch_size=batch_size)
        self.raw_data = {}
        self.embeddings = None
        self.article_keys = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_all_embeddings_as_database(self):
        try:
            print("Đang tải database...")
            
            # 1. Tải tất cả dữ liệu JSON
            from_json_path = 'data/processed'
            all_raw_data = {}
            
            if not os.path.exists(from_json_path):
                raise FileNotFoundError(f"Thư mục {from_json_path} không tồn tại")
            
            json_files = [f for f in os.listdir(from_json_path) if f.endswith('.json')]
            print(f"Tìm thấy {len(json_files)} file JSON")
            
            for file_name in json_files:
                file_path = os.path.join(from_json_path, file_name)
                file_prefix = os.path.splitext(file_name)[0]
                
                print(f"Đang tải {file_name}...")
                article_data = self.embed.load_article_data(file_path)
                
                # Thêm prefix để tránh trung key giữa các file
                for key, value in article_data.items():
                    prefixed_key = f"{file_prefix.strip().lower()}_{key}"
                    all_raw_data[prefixed_key] = value
            
            self.raw_data = all_raw_data
            print(f"Đã tải tổng cong {len(all_raw_data)} điều luật")
            
            # 2. Tải tất cả embeddings
            from_embed_path = 'data/embeddings'
            all_embeddings = []
            all_article_keys = []
            
            if not os.path.exists(from_embed_path):
                raise FileNotFoundError(f"Thư mục {from_embed_path} không tồn tại")
            
            embed_files = [f for f in os.listdir(from_embed_path) if f.endswith('.pkl')]
            print(f"Tìm thấy {len(embed_files)} file embeddings")
            
            for file_name in embed_files:
                file_path = os.path.join(from_embed_path, file_name)
                file_prefix = os.path.splitext(file_name)[0]
                
                print(f"Đang tải {file_name}...")
                embeddings, article_keys = self.embed.load_embeddings(file_path)
                
                # Thêm prefix cho article keys
                prefixed_keys = [f"{file_prefix.strip().lower()}_{key}" for key in article_keys]
                
                all_embeddings.append(embeddings)
                all_article_keys.extend(prefixed_keys)
            
            # Kết hợp tất cả embeddings
            if all_embeddings:
                self.embeddings = np.vstack(all_embeddings)
                self.article_keys = all_article_keys
                
                print(f"Database đã sẵn sàng:")
                print(f"  - Tổng số điều luật: {len(self.raw_data)}")
                print(f"  - Shape embeddings: {self.embeddings.shape}")
                print(f"  - Số lượng keys: {len(self.article_keys)}")
                
                # Kiểm tra tính nhất quán
                if len(self.article_keys) != self.embeddings.shape[0]:
                    print("Cảnh báo: Số lượng keys không khớp với số lượng embeddings")
                    
            else:
                raise ValueError("Không tìm thấy embeddings nào")
                
        except Exception as e:
            print(f"Lỗi khi tải database: {e}")
            raise
    
    def preprocess_text_for_embedding(self, text: str) -> str:
        text = re.sub(r'[^\w\s\.,;:\-\(\)\[\]\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        return ' '.join(filtered_tokens)
    
    def create_embedding(self, text: str) -> np.ndarray:
        processed_text = self.preprocess_text_for_embedding(text)
        return self.embed.create_embedding_single(processed_text)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm tối ưu sử dụng GPU (nếu có) cho việc tính cosine similarity
        """
        if self.embeddings is None:
            raise ValueError("Chưa tạo embeddings cho dữ liệu. Hãy chạy load_all_embeddings_as_database() trước.")
        
        query_embedding = self.create_embedding(query)
        
        if torch.cuda.is_available():
            # Sử dụng GPU để tính similarity
            query_tensor = torch.tensor(query_embedding, device=self.device).unsqueeze(0)
            embeddings_tensor = torch.tensor(self.embeddings, device=self.device)
            
            # Tính cosine similarity trên GPU
            similarities = torch.nn.functional.cosine_similarity(
                query_tensor, embeddings_tensor, dim=1
            ).cpu().numpy()
        else:
            # Fallback về CPU
            query_embedding = query_embedding.reshape(1, -1)
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            article_key = self.article_keys[idx]
            
            base_key = "_".join(article_key.split("_")[:-2])
            
            if base_key in self.raw_data:
                article_data = self.raw_data[base_key]
                
                result = {
                    'article_key': article_key,
                    'base_article_key': base_key,
                    'title': article_data['title'],
                    'content': article_data['text'],
                    'similarity_score': float(similarities[idx]),
                    'rank': len(results) + 1
                }
                results.append(result)
            else:
                print(f"Cảnh báo: Không tìm thấy dữ liệu cho key {base_key}")
        
        return results
    

def main():
    # Tăng batch_size nếu có GPU
    batch_size = 64 if torch.cuda.is_available() else 32
    search_system = SearchEngine(batch_size=batch_size)
    
    try:
        search_system.load_all_embeddings_as_database()
        
        sample_query = "Người sử dụng lao động được sa thải người lao động nữ đang mang thai không?"
        results = search_system.search(sample_query, top_k=5)
        
        print(results)
        
    except Exception as e:
        print(f"Lỗi trong quá trình chạy: {e}")

if __name__ == "__main__":
    main()