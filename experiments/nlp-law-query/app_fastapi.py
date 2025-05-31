from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import time
from contextlib import asynccontextmanager
import os
import sys
import torch

try:
    from src.search_engine import SearchEngine
    from src.stop_word import download_vietnamese_stopwords
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.search_engine import SearchEngine
    from src.stop_word import download_vietnamese_stopwords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

search_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine
    
    # Startup
    logger.info("Khởi tạo Search Engine...")
    try:
        download_vietnamese_stopwords()
        batch_size = 64 if torch.cuda.is_available() else 32
        search_engine = SearchEngine(batch_size=batch_size)
        
        # Load database
        logger.info("Đang tải database...")
        search_engine.load_all_embeddings_as_database()
        logger.info("Search Engine đã sẵn sàng!")
        
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo Search Engine: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Đang tắt Search Engine...")
    search_engine = None


app = FastAPI(
    title="Legal Document Search API",
    description="API tìm kiếm văn bản pháp luật sử dụng AI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Câu hỏi tìm kiếm")
    top_k: Optional[int] = Field(default=5, ge=1, le=50, description="Số lượng kết quả trả về")

class SearchResult(BaseModel):
    article_key: str
    base_article_key: str
    title: str
    content: str
    similarity_score: float
    rank: int

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    gpu_available: bool
    model_loaded: bool


@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Legal Document Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    global search_engine
    
    return HealthResponse(
        status="healthy" if search_engine is not None else "unhealthy",
        message="Search Engine đã sẵn sàng" if search_engine is not None else "Search Engine chưa được khởi tạo",
        gpu_available=torch.cuda.is_available(),
        model_loaded=search_engine is not None
    )

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Tìm kiếm văn bản pháp luật
    
    - **query**: Câu hỏi tìm kiếm
    - **top_k**: Số lượng kết quả trả về (1-50)
    """
    global search_engine
    
    if search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search Engine chưa được khởi tạo. Vui lòng thử lại sau."
        )
    
    try:
        start_time = time.time()
        
        results = search_engine.search(request.query, top_k=request.top_k)
        
        processing_time = time.time() - start_time
        
        search_results = [
            SearchResult(
                article_key=result['article_key'],
                base_article_key=result['base_article_key'],
                title=result['title'],
                content=result['content'],
                similarity_score=result['similarity_score'],
                rank=result['rank']
            )
            for result in results
        ]
        
        logger.info(f"Tìm kiếm thành công: '{request.query}' - {len(results)} kết quả - {processing_time:.3f}s")
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(results),
            processing_time=processing_time,
            message=f"Tìm thấy {len(results)} kết quả phù hợp"
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tìm kiếm: {str(e)}"
        )
if __name__ == "__main__":
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )