import json
import re
from typing import Dict, List, Tuple
import numpy as np
import os
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from underthesea import word_tokenize, sent_tokenize

class Embed():
    def __init__(self, model_name: str = "vinai/phobert-base-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.embed_path = os.path.join("data", 'embeddings')
        os.makedirs(self.embed_path, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Sử dụng device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        print(f"Đang tải model {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Tải model thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model '{self.model_name}': {e}")
            raise e
    
    def create_embedding_single(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy().flatten()
    
    def create_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """Tạo embedding cho nhiều text cùng lúc"""
        if not texts:
            return np.array([])
        
        inputs = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Chuyển inputs lên device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy()
    
    def load_article_data(self, json_path: str) -> Dict[str, Dict[str, str]]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                legal_data = json.load(f)
            print(f"Đã tải {len(legal_data)} điều luật từ {json_path}")
            return legal_data
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file tại đường dẫn {json_path}")
            raise
        except json.JSONDecodeError:
            print(f"Lỗi: File {json_path} không phải là định dạng JSON hợp lệ.")
            raise
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu từ {json_path}: {e}")
            raise
    
    def split_to_chunks(self, text: str) -> List[str]:
        section_patterns = [
            r'(\d+)\.\s+(.*?)(?=(\d+\.\s)|$)',
            r'([a-z])\)\s+(.*?)(?=([a-z]\)\s)|$)',
            r'([IVX]+)\.\s+(.*?)(?=([IVX]+\.\s)|$)'
        ]
        chunks = sent_tokenize(text)
        if not chunks:
            # Nếu không match gì: fallback sang pattern regex split
            chunks = []
            for pattern in section_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    chunk_content = match.group(2).strip()
                    if chunk_content:
                        chunks.append(chunk_content)

        if not chunks:
            print("Không tìm thấy bất kỳ đoạn văn nào trong văn bản. Sử dụng toàn bộ văn bản làm một chunk.")
            chunks = [text.strip()]
        return chunks
    
    def create_embeddings_for_article(self, file_path: str):
        """Phiên bản tối ưu sử dụng batch processing"""
        if not file_path:
            print("File path is empty")
            raise ValueError("File path cannot be empty")
        
        file_name = os.path.basename(file_path)
        file_to_save = os.path.join(self.embed_path, f'{os.path.splitext(file_name)[0]}.pkl')
        
        if os.path.exists(file_to_save):
            print(f"File embeddings đã tồn tại tại {file_to_save}, bỏ qua việc tạo mới.")
            return
        
        print(f"Đang tạo embeddings cho điều luật {file_path}...")
        
        legal_data = self.load_article_data(file_path)
        
        all_texts = []
        all_chunk_ids = []
        
        for article_key, article_data in legal_data.items():
            title = article_data.get('title', '')
            text = article_data.get('text', '')
            
            if not text:
                print(f"Cảnh báo: Điều luật '{article_key}' không có nội dung, bỏ qua.")
                continue
                
            chunks = self.split_to_chunks(text)
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{article_key}_chunk_{idx}"
                combined_text = f"{title} {chunk}"
                
                all_texts.append(combined_text)
                all_chunk_ids.append(chunk_id)
        
        print(f"Tổng số chunks cần xử lý: {len(all_texts)}")
        
        # Xử lý theo batch
        all_embeddings = []
        
        for i in tqdm(range(0, len(all_texts), self.batch_size), desc="Tạo embeddings (batch)"):
            batch_texts = all_texts[i:i + self.batch_size]
            batch_embeddings = self.create_embedding_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Kết hợp tất cả embeddings
        embeddings = np.vstack(all_embeddings)
        
        print(f"Đã tạo embeddings cho {len(all_texts)} chunks")
        print(f"Shape của embeddings: {embeddings.shape}")
        
        self.save_embeddings_to_pkl(all_chunk_ids, embeddings, file_to_save)
    
    def save_embeddings_to_pkl(self, article_keys: List[str], embeddings: np.ndarray, save_path: str):
        embedding_data = {
            'embeddings': embeddings,
            'article_keys': article_keys,
            'model_name': self.model_name,
            'batch_size': self.batch_size
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"Đã lưu embeddings vào {save_path}")
    
    def load_embeddings(self, load_path: str) -> Tuple[np.ndarray, List[str]]:
        try:
            with open(load_path, 'rb') as f:
                embedding_data = pickle.load(f)
            
            embeddings = embedding_data['embeddings']
            article_keys = embedding_data['article_keys']
            
            print(f"Đã tải embeddings từ {load_path}")
            print(f"Shape của embeddings: {embeddings.shape}")
            
            return embeddings, article_keys
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file embeddings tại đường dẫn {load_path}")
            raise
        except Exception as e:
            print(f"Lỗi khi tải embeddings từ {load_path}: {e}")
            raise

if __name__ == '__main__':
    from_path = 'data/processed'
    list_files = os.listdir(from_path)
    
    batch_size = 64 if torch.cuda.is_available() else 32
    embed = Embed(batch_size=batch_size)
    
    for file_name in list_files:
        if file_name.endswith('.json'):
            file_path = os.path.join(from_path, file_name)
            embed.create_embeddings_for_article(file_path)