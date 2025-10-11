import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.index = None
        self.documents = []
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """인지적 오류 문서를 로드하고 FAISS 인덱스를 생성합니다."""
        try:
            # 인지적 오류 문서 로드
            cognitive_errors_doc = self._load_cognitive_errors_document()
            if not cognitive_errors_doc:
                logger.error("인지적 오류 문서가 비어있습니다.")
                return
            
            # 문서를 청크로 분할
            self.documents = self._split_document_into_chunks(cognitive_errors_doc)
            logger.info(f"문서 청크 분할 완료: {len(self.documents)}개 청크")
            
            # 임베딩 생성
            logger.info("임베딩 생성 시작...")
            embeddings = self.embeddings.embed_documents(self.documents)
            embeddings_array = np.array(embeddings).astype('float32')
            logger.info(f"임베딩 생성 완료: {embeddings_array.shape}")
            
            # FAISS 인덱스 생성
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # 정규화 (cosine similarity를 위해)
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            logger.info(f"FAISS 인덱스 생성 완료: {len(self.documents)}개 문서")
            
        except Exception as e:
            logger.error(f"FAISS 인덱스 생성 중 오류: {e}")
            self.index = None
            self.documents = []
    
    def _load_cognitive_errors_document(self) -> str:
        """인지적 오류 문서를 로드합니다."""
        try:
            json_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'cognitive_errors.json')
            logger.info(f"인지적 오류 JSON 문서 경로: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON을 텍스트로 변환
                content = ""
                for error in data.get('cognitive_errors', []):
                    content += f"{error['error_type']}\n"
                    content += f"{error['description']}\n"
                    content += "\n"
                logger.info(f"인지적 오류 JSON 문서 로드 성공: {len(content)}자")
                return content
        except Exception as e:
            logger.error(f"인지적 오류 JSON 문서 로드 실패: {e}")
            return ""
    
    def _split_document_into_chunks(self, document: str) -> List[str]:
        """문서를 의미있는 청크로 분할합니다."""
        chunks = []
        lines = document.split('\n')
        
        current_chunk = ""
        for line in lines:
            line = line.strip()
            if not line:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                continue
            
            # 새로운 인지적 오류 유형이 시작되는 경우
            if any(keyword in line for keyword in ['이분법적 사고', '완벽주의', '의미 확대', '과잉일반화', 
                                                 '정신적 여과', '잘못된 명명', '예언자적 오류', 
                                                 '허황된 인과관계', '감정적 추리']):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += " " + line
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def search_relevant_cognitive_errors(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """쿼리와 관련된 인지적 오류를 검색합니다."""
        if not self.index or not self.documents:
            logger.warning("FAISS 인덱스가 없습니다. 인덱스를 생성합니다.")
            self._load_or_create_index()
            if not self.index or not self.documents:
                logger.error("인덱스 생성에 실패했습니다.")
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)
            
            # 검색 수행
            scores, indices = self.index.search(query_vector, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"인지적 오류 검색 중 오류: {e}")
            return []
    
    def extract_cognitive_errors_from_context(self, context: str) -> List[Dict[str, str]]:
        """컨텍스트에서 인지적 오류를 추출합니다."""
        logger.info(f"컨텍스트에서 인지적 오류 추출 시작: {context[:100]}...")
        
        # JSON 파일에서 직접 인지적 오류 정보 로드
        try:
            json_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'cognitive_errors.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_errors = data.get('cognitive_errors', [])
                logger.info(f"JSON에서 로드된 인지적 오류 수: {len(all_errors)}")
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패: {e}")
            return []
        
        # 검색된 관련 문서들을 기반으로 인지적 오류 추출
        relevant_docs = self.search_relevant_cognitive_errors(context, top_k=5)
        logger.info(f"검색된 관련 문서 수: {len(relevant_docs)}")
        
        cognitive_errors = []
        found_error_types = set()
        
        # 검색된 문서에서 인지적 오류 타입 찾기
        for doc in relevant_docs:
            logger.info(f"문서 검토 중: {doc['document'][:100]}...")
            for error in all_errors:
                error_type = error['error_type']
                if error_type in doc['document'] and error_type not in found_error_types:
                    logger.info(f"인지적 오류 발견: {error_type}")
                    cognitive_errors.append({
                        'error_type': error_type,
                        'description': error['description']
                    })
                    found_error_types.add(error_type)
                    logger.info(f"인지적 오류 추가: {error_type}")
        
        logger.info(f"최종 추출된 인지적 오류 수: {len(cognitive_errors)}")
        return cognitive_errors[:3]  # 최대 3개 반환
