import os
import time
import json
import hashlib
import pickle
import uuid
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import gradio as gr
from dotenv import load_dotenv
import numpy as np

# --- Core Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# --- Advanced RAG Imports ---
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MultiQueryRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.stores import InMemoryStore

try:
    from langchain_community.document_compressors import FlashrankRerank
    FlashrankRerank.model_rebuild()
except ImportError:
    pass

load_dotenv()

# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

class RAGConfig:
    """Configuration for RAG system"""
    CHUNK_STRATEGIES = {
        "balanced": {"size": 700, "overlap": 120},
        "precise": {"size": 400, "overlap": 80},
        "contextual": {"size": 1200, "overlap": 200},
        "hierarchical": {"parent_size": 2000, "child_size": 500, "overlap": 100}
    }
    
    RETRIEVAL_STRATEGIES = {
        "hybrid": {"bm25_weight": 0.5, "vector_weight": 0.5},
        "semantic": {"bm25_weight": 0.3, "vector_weight": 0.7},
        "keyword": {"bm25_weight": 0.7, "vector_weight": 0.3}
    }
    
    EMBEDDING_MODELS = {
        "fast": "all-MiniLM-L6-v2",
        "balanced": "sentence-transformers/all-mpnet-base-v2",
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }
    
    API_RETRY_DELAY = 2
    MAX_API_RETRIES = 3

class SemanticCache:
    """Cache for similar queries"""
    def __init__(self, threshold: float = 0.85):
        self.cache = {}
        self.embeddings = None
        self.threshold = threshold
    
    def _get_key(self, query: str, embeddings) -> str:
        if self.embeddings is None:
            self.embeddings = embeddings
        
        query_embedding = self.embeddings.embed_query(query)
        
        for cached_query, (cached_emb, result) in self.cache.items():
            similarity = np.dot(query_embedding, cached_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_emb)
            )
            if similarity > self.threshold:
                return cached_query
        
        return query
    
    def get(self, query: str, embeddings) -> Optional[Any]:
        key = self._get_key(query, embeddings)
        return self.cache.get(key)
    
    def set(self, query: str, result: Any, embeddings):
        if self.embeddings is None:
            self.embeddings = embeddings
        query_embedding = self.embeddings.embed_query(query)
        self.cache[query] = (query_embedding, result)

# ============================================================================
# THREAD MANAGEMENT
# ============================================================================

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create the chat history container for a specific thread."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

class ThreadData:
    """Container for thread-specific data including documents, vector store, and chain."""
    def __init__(self, thread_id: str, thread_name: str):
        self.thread_id = thread_id
        self.thread_name = thread_name
        self.created_at = datetime.now().isoformat()
        self.processed_files = []
        self.vector_store = None
        self.chain = None
        self.splits = None
        self.parent_store = InMemoryStore()
        self.persist_dir = f"./chroma_db_threads/{thread_id}"
        self.document_analyses = {}
        self.query_log = []
        self.stats = {
            "total_chunks": 0,
            "total_pages": 0,
            "processing_time": 0,
            "queries_count": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
        os.makedirs(self.persist_dir, exist_ok=True)

# ============================================================================
# DOCUMENT PROCESSING & ANALYSIS
# ============================================================================

class DocumentAnalyzer:
    """Advanced document analysis and metadata extraction"""
    
    @staticmethod
    def extract_metadata(text: str, page_num: int, doc_name: str) -> Dict[str, Any]:
        metadata = {
            "page": page_num,
            "source": doc_name,
            "chunk_size": len(text.split()),
            "char_count": len(text)
        }
        
        text_lower = text.lower()
        
        section_keywords = {
            "abstract": ["abstract", "summary", "overview"],
            "introduction": ["introduction", "background", "motivation"],
            "methodology": ["method", "methodology", "approach", "procedure"],
            "results": ["result", "finding", "outcome", "analysis"],
            "discussion": ["discussion", "implication", "interpretation"],
            "conclusion": ["conclusion", "summary", "future work"],
            "references": ["reference", "bibliography", "citation"]
        }
        
        for section, keywords in section_keywords.items():
            if any(kw in text_lower[:150] for kw in keywords):
                metadata["section"] = section
                break
        else:
            metadata["section"] = "general"
        
        if any(indicator in text for indicator in ["Figure", "Table", "Algorithm"]):
            metadata["contains_visual"] = True
        
        words = text_lower.split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 4:
                word_freq[word] += 1
        
        top_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        metadata["key_terms"] = ", ".join([term for term, _ in top_terms])
        
        return metadata
    
    @staticmethod
    def analyze_document_structure(docs: List[Document]) -> Dict[str, Any]:
        analysis = {
            "total_pages": len(docs),
            "estimated_word_count": sum(len(doc.page_content.split()) for doc in docs),
            "sections_found": {},
            "avg_page_length": 0
        }
        
        section_counts = defaultdict(int)
        for doc in docs:
            section = doc.metadata.get("section", "general")
            section_counts[section] += 1
        
        analysis["sections_found"] = dict(section_counts)
        analysis["avg_page_length"] = analysis["estimated_word_count"] / len(docs) if docs else 0
        
        return analysis

# ============================================================================
# ADVANCED RETRIEVAL STRATEGIES
# ============================================================================

class AdvancedRetriever:
    """Implements multiple advanced retrieval strategies"""
    
    def __init__(self, vector_store, splits, llm):
        self.vector_store = vector_store
        self.splits = splits
        self.llm = llm
    
    def create_hyde_retriever(self, k: int = 5):
        hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Please write a scientific passage to answer the question
Question: {question}
Passage:"""
        )
        
        hyde_chain = hyde_prompt | self.llm | StrOutputParser()
        return hyde_chain
    
    def create_multi_query_retriever(self):
        return MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(),
            llm=self.llm
        )
    
    def create_ensemble_retriever(self, strategy: str = "hybrid"):
        config = RAGConfig.RETRIEVAL_STRATEGIES[strategy]
        
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 40}
        )
        
        bm25_retriever = BM25Retriever.from_documents(self.splits)
        bm25_retriever.k = 10
        
        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[config["bm25_weight"], config["vector_weight"]]
        )
    
    def create_contextual_compression_retriever(self, base_retriever):
        compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=5
        )
        
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

# ============================================================================
# SELF-RAG: QUERY ANALYSIS & ANSWER VERIFICATION
# ============================================================================

class SelfRAG:
    """Self-reflective RAG with query routing and answer verification"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        analysis_prompt = f"""Analyze this query briefly and respond with JSON:
Query: {query}

JSON format: {{"complexity": "simple/moderate/complex", "query_type": "factual/analytical/comparative/exploratory", "requires_multiple_sources": true/false, "suggested_retrieval_k": 5-10}}

Respond ONLY with valid JSON, no explanation."""
        
        try:
            time.sleep(RAGConfig.API_RETRY_DELAY)
            response = self.llm.invoke(analysis_prompt)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ Query analysis failed: {str(e)}")
            return {
                "complexity": "moderate",
                "query_type": "factual",
                "requires_multiple_sources": True,
                "suggested_retrieval_k": 8
            }
    
    def verify_answer(self, question: str, answer: str, context: List[Document]) -> Dict[str, Any]:
        verification_prompt = f"""Evaluate briefly:
Question: {question}
Answer: {answer[:500]}...

Rate 1-10 as JSON: {{"relevance": X, "support": X, "completeness": X, "confidence": X, "needs_refinement": true/false}}"""
        
        try:
            time.sleep(RAGConfig.API_RETRY_DELAY)
            response = self.llm.invoke(verification_prompt)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ Answer verification failed: {str(e)}")
            return {
                "relevance": 7,
                "support": 7,
                "completeness": 7,
                "confidence": 7,
                "needs_refinement": False
            }

# ============================================================================
# MAIN RAG SYSTEM WITH THREAD MANAGEMENT
# ============================================================================

class UltimateRAGSystem:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.threads: Dict[str, ThreadData] = {}
        self.semantic_cache = SemanticCache()
        self.self_rag = None
        self.global_stats = {
            "total_threads": 0,
            "total_documents_processed": 0
        }
    
    def create_new_thread(self, thread_name: Optional[str] = None) -> str:
        """Create a new conversation thread with its own isolated document collection."""
        thread_id = str(uuid.uuid4())
        name = thread_name or f"Thread {len(self.threads) + 1}"
        
        self.threads[thread_id] = ThreadData(thread_id, name)
        get_session_history(thread_id)
        
        self.global_stats["total_threads"] += 1
        
        return thread_id
    
    def get_thread_info(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific thread."""
        if thread_id not in self.threads:
            return None
        
        thread = self.threads[thread_id]
        history = get_session_history(thread_id)
        
        return {
            "name": thread.thread_name,
            "created_at": thread.created_at,
            "message_count": len(history.messages),
            "document_count": len(thread.processed_files),
            "documents": thread.processed_files,
            "total_chunks": thread.stats["total_chunks"],
            "total_pages": thread.stats["total_pages"]
        }
    
    def list_threads(self) -> List[Dict[str, Any]]:
        """List all threads with their metadata."""
        threads = []
        for thread_id in self.threads:
            info = self.get_thread_info(thread_id)
            if info:
                threads.append({"id": thread_id, **info})
        return threads
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its associated data."""
        if thread_id not in self.threads:
            return False
        
        thread = self.threads[thread_id]
        
        if os.path.exists(thread.persist_dir):
            shutil.rmtree(thread.persist_dir)
        
        if thread_id in store:
            del store[thread_id]
        
        del self.threads[thread_id]
        
        return True
    
    def process_document(
        self,
        file,
        thread_id: str,
        chunk_strategy: str = "balanced",
        embedding_model: str = "fast",
        enable_hierarchical: bool = True
    ):
        """Process document and add to specific thread's collection"""
        if not file:
            return "⚠️ Please upload a file.", self._get_thread_stats_display(thread_id), ""
        
        if thread_id not in self.threads:
            return "⚠️ Invalid thread ID.", self._get_thread_stats_display(thread_id), ""
        
        thread = self.threads[thread_id]
        start_time = time.time()
        
        try:
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext == '.pdf':
                loader = PyPDFLoader(file.name)
            elif file_ext == '.txt':
                loader = TextLoader(file.name)
            elif file_ext in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file.name)
            else:
                return f"❌ Unsupported file type: {file_ext}", self._get_thread_stats_display(thread_id), ""
            
            docs = loader.load()
            file_name = os.path.basename(file.name)
            
            if file_name in thread.processed_files:
                return f"⚠️ {file_name} already processed in this thread.", self._get_thread_stats_display(thread_id), ""
            
            doc_analysis = DocumentAnalyzer.analyze_document_structure(docs)
            thread.document_analyses[file_name] = doc_analysis
            
            for idx, doc in enumerate(docs):
                doc.metadata["source"] = file_name
                doc.metadata["thread_id"] = thread_id
                doc.metadata.update(
                    DocumentAnalyzer.extract_metadata(doc.page_content, idx, file_name)
                )
            
            if enable_hierarchical:
                new_splits = self._create_hierarchical_chunks(docs, thread, chunk_strategy)
                strategy_name = f"Hierarchical-{chunk_strategy}"
            else:
                config = RAGConfig.CHUNK_STRATEGIES[chunk_strategy]
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config["size"],
                    chunk_overlap=config["overlap"],
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                new_splits = text_splitter.split_documents(docs)
                strategy_name = chunk_strategy.title()
            
            if not self.embeddings:
                model_name = RAGConfig.EMBEDDING_MODELS[embedding_model]
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            if thread.vector_store is None:
                thread.vector_store = Chroma.from_documents(
                    documents=new_splits,
                    embedding=self.embeddings,
                    persist_directory=thread.persist_dir
                )
                thread.splits = new_splits
            else:
                thread.vector_store.add_documents(new_splits)
                thread.splits.extend(new_splits)
            
            thread.vector_store.persist()
            
            self._initialize_components(thread_id)
            
            thread.processed_files.append(file_name)
            thread.stats["total_chunks"] = len(thread.splits)
            thread.stats["total_pages"] += len(docs)
            thread.stats["processing_time"] = round(time.time() - start_time, 2)
            
            self.global_stats["total_documents_processed"] += 1
            
            analysis_text = self._format_document_analysis(doc_analysis)
            
            return (
                f"✅ {strategy_name} | Added {file_name}: {len(new_splits)} chunks from {len(docs)} pages\n"
                f"📚 Thread now has {len(thread.processed_files)} document(s) with {thread.stats['total_chunks']} total chunks",
                self._get_thread_stats_display(thread_id),
                analysis_text
            )
            
        except Exception as e:
            return f"❌ Error: {str(e)}", self._get_thread_stats_display(thread_id), ""
    
    def _create_hierarchical_chunks(self, docs: List[Document], thread: ThreadData, strategy: str) -> List[Document]:
        config = RAGConfig.CHUNK_STRATEGIES["hierarchical"]
        
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["parent_size"],
            chunk_overlap=config["overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["child_size"],
            chunk_overlap=config["overlap"] // 2,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        parent_docs = parent_splitter.split_documents(docs)
        child_docs = []
        
        for parent_idx, parent_doc in enumerate(parent_docs):
            parent_id = f"parent_{parent_idx}_{uuid.uuid4().hex[:8]}"
            children = child_splitter.split_documents([parent_doc])
            for child in children:
                child.metadata["parent_id"] = parent_id
            child_docs.extend(children)
            thread.parent_store.mset([(parent_id, parent_doc)])
        
        return child_docs
    
    def _initialize_components(self, thread_id: str):
        """Initialize LLM and RAG components for a specific thread"""
        if thread_id not in self.threads:
            return
        
        thread = self.threads[thread_id]
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY")
        
        if not self.llm:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                temperature=0.2,
                google_api_key=api_key
            )
        
        if not self.self_rag:
            self.self_rag = SelfRAG(self.llm)
        
        self._build_chain(thread_id)
    
    def _build_chain(self, thread_id: str, retrieval_strategy: str = "hybrid"):
        """Build RAG chain for a specific thread"""
        if thread_id not in self.threads:
            return
        
        thread = self.threads[thread_id]
        
        advanced_retriever = AdvancedRetriever(thread.vector_store, thread.splits, self.llm)
        ensemble_retriever = advanced_retriever.create_ensemble_retriever(retrieval_strategy)
        compression_retriever = advanced_retriever.create_contextual_compression_retriever(ensemble_retriever)
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            (MessagesPlaceholder("chat_history")),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, compression_retriever, contextualize_q_prompt
        )
        
        qa_system_prompt = """You are an expert research assistant. Use the retrieved context to provide accurate, well-structured answers.

Guidelines:
- Be precise and cite specific information from the context
- If information is insufficient, acknowledge limitations
- Structure complex answers with clear organization
- Highlight key insights and connections

Context:
{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            (MessagesPlaceholder("chat_history")),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        thread.chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def chat(
        self,
        message: str,
        thread_id: str,
        history,
        use_self_rag: bool = False,
        use_cache: bool = True,
        retrieval_strategy: str = "hybrid",
        temperature: float = 0.2
    ):
        """Advanced chat with thread isolation"""
        if thread_id not in self.threads:
            return "⚠️ Invalid thread ID."
        
        thread = self.threads[thread_id]
        
        if not thread.chain:
            return "⚠️ Please upload and process a document first for this thread."
        
        query_start = time.time()
        
        if use_cache:
            cached = self.semantic_cache.get(message, self.embeddings)
            if cached:
                thread.stats["cache_hits"] += 1
                return cached + "\n\n💾 *[Retrieved from cache]*"
        
        if use_self_rag:
            print("🔍 Analyzing query...")
            query_analysis = self.self_rag.analyze_query(message)
        else:
            query_analysis = {"complexity": "moderate"}
        
        self.llm.temperature = temperature
        
        if retrieval_strategy != "hybrid":
            self._build_chain(thread_id, retrieval_strategy)
        
        config = {"configurable": {"session_id": thread_id}}
        
        max_retries = RAGConfig.MAX_API_RETRIES
        retry_delay = RAGConfig.API_RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⏳ Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                
                response = thread.chain.invoke({"input": message}, config=config)
                answer = response["answer"]
                context_docs = response["context"]
                
                if use_self_rag:
                    print("✅ Verifying answer...")
                    verification = self.self_rag.verify_answer(message, answer, context_docs)
                    confidence_score = np.mean([
                        verification["relevance"],
                        verification["support"],
                        verification["completeness"],
                        verification["confidence"]
                    ]) * 10
                else:
                    confidence_score = 75
                
                sources_detail = self._format_sources(context_docs, query_analysis)
                
                if confidence_score >= 80:
                    conf_emoji = "🟢"
                elif confidence_score >= 60:
                    conf_emoji = "🟡"
                else:
                    conf_emoji = "🔴"
                
                metadata = f"\n\n{conf_emoji} **Confidence:** {confidence_score:.0f}%"
                
                if use_self_rag:
                    metadata += f" | **Complexity:** {query_analysis.get('complexity', 'N/A').title()}"
                
                final_answer = answer + sources_detail + metadata
                
                if use_cache:
                    self.semantic_cache.set(message, final_answer, self.embeddings)
                
                response_time = time.time() - query_start
                thread.stats["queries_count"] += 1
                thread.stats["total_response_time"] += response_time
                thread.stats["avg_response_time"] = thread.stats["total_response_time"] / thread.stats["queries_count"]
                
                thread.query_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": message,
                    "confidence": confidence_score,
                    "response_time": response_time,
                    "sources_count": len(context_docs)
                })
                
                return final_answer
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️ API quota exceeded. Retrying...")
                        continue
                    else:
                        return "❌ API Quota Exhausted. Please wait or upgrade your plan.\n\n💡 **Tips:**\n- Wait ~1 hour for quota reset\n- Disable Self-RAG to reduce API calls\n- Use semantic cache to avoid repeated queries\n- Consider upgrading to a paid plan"
                else:
                    return f"❌ Error: {error_msg}"
        
        return "❌ Maximum retries exceeded. Please try again later."
    
    def _format_sources(self, docs: List[Document], query_analysis: Dict) -> str:
        if not docs:
            return ""
        
        sources = []
        seen = set()
        
        for doc in docs[:5]:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section", "general")
            
            key = f"{source}_{page}_{section}"
            if key not in seen:
                preview = doc.page_content[:100].replace("\n", " ") + "..."
                sources.append(
                    f"📄 **{source}** | Page {page} | {section.title()}\n   _{preview}_"
                )
                seen.add(key)
        
        return "\n\n**📚 Sources:**\n" + "\n\n".join(sources)
    
    def _get_thread_stats_display(self, thread_id: str) -> str:
        if thread_id not in self.threads:
            return "⚠️ Invalid thread"
        
        thread = self.threads[thread_id]
        
        return f"""
📊 **Thread Statistics: {thread.thread_name}**

**Documents:**
- 📚 Files in Thread: {len(thread.processed_files)}
- 📄 Total Pages: {thread.stats['total_pages']}
- 🧩 Total Chunks: {thread.stats['total_chunks']}

**Performance:**
- ⏱️ Last Processing: {thread.stats['processing_time']}s
- 💬 Queries Handled: {thread.stats['queries_count']}
- 💾 Cache Hits: {thread.stats['cache_hits']}
- ⚡ Avg Response Time: {thread.stats['avg_response_time']:.2f}s

**📄 Documents:**
{chr(10).join(f"  • {doc}" for doc in thread.processed_files) if thread.processed_files else "  None yet"}
"""
    
    def _get_global_stats_display(self) -> str:
        return f"""
🌍 **Global Statistics**
- 🧵 Total Threads: {len(self.threads)}
- 📚 Total Documents Processed: {self.global_stats['total_documents_processed']}
"""
    
    def _format_document_analysis(self, analysis: Dict) -> str:
        sections = "\n".join([f"- {k.title()}: {v}" for k, v in analysis["sections_found"].items()])
        
        return f"""
📖 **Document Analysis**

**Structure:**
- Total Pages: {analysis['total_pages']}
- Estimated Words: {analysis['estimated_word_count']:,}
- Avg Page Length: {analysis['avg_page_length']:.0f} words

**Sections Found:**
{sections}
"""
    
    def export_thread(self, thread_id: str) -> str:
        """Export a specific thread's conversation to JSON."""
        if thread_id not in self.threads:
            return None
        
        thread = self.threads[thread_id]
        history = get_session_history(thread_id)
        messages = []
        
        for msg in history.messages:
            messages.append({
                "role": msg.type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "thread_id": thread_id,
            "thread_name": thread.thread_name,
            "created_at": thread.created_at,
            "documents": thread.processed_files,
            "stats": thread.stats,
            "conversation": messages
        }
        
        filename = f"rag_thread_{thread_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def export_all_threads(self) -> str:
        """Export all threads to a single JSON file."""
        all_threads = {}
        
        for thread_id, thread in self.threads.items():
            history = get_session_history(thread_id)
            messages = []
            
            for msg in history.messages:
                messages.append({
                    "role": msg.type,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })
            
            all_threads[thread_id] = {
                "name": thread.thread_name,
                "created_at": thread.created_at,
                "documents": thread.processed_files,
                "stats": thread.stats,
                "conversation": messages
            }
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_threads": len(all_threads),
            "global_stats": self.global_stats,
            "threads": all_threads
        }
        
        filename = f"rag_all_threads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def get_query_analytics(self, thread_id: str) -> str:
        """Generate query analytics for a thread"""
        if thread_id not in self.threads:
            return "⚠️ Invalid thread"
        
        thread = self.threads[thread_id]
        
        if not thread.query_log:
            return "No queries yet."
        
        avg_conf = np.mean([q["confidence"] for q in thread.query_log])
        avg_time = np.mean([q["response_time"] for q in thread.query_log])
        
        recent = thread.query_log[-5:]
        recent_queries = "\n".join([
            f"- {q['query'][:50]}... (Conf: {q['confidence']:.0f}%)"
            for q in recent
        ])
        
        return f"""
📈 **Query Analytics**

**Overall:**
- Total Queries: {len(thread.query_log)}
- Avg Confidence: {avg_conf:.1f}%
- Avg Response Time: {avg_time:.2f}s

**Recent Queries:**
{recent_queries}
"""

# ============================================================================
# GRADIO INTERFACE WITH THREAD MANAGEMENT
# ============================================================================

rag_system = UltimateRAGSystem()
default_thread_id = rag_system.create_new_thread("Default Conversation")

with gr.Blocks(title="Ultimate RAG System with Threads", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Ultimate Advanced RAG System with Thread Management
    ### Multi-Strategy Retrieval | Self-RAG | Semantic Caching | Isolated Thread Collections
    """)
    
    current_thread = gr.State(value=default_thread_id)
    
    with gr.Tabs():
        # ==================== TAB 1: DOCUMENT UPLOAD ====================
        with gr.Tab("📤 Document Processing"):
            gr.Markdown("### Upload and Configure Document Processing for Current Thread")
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".txt", ".doc", ".docx"]
                    )
                    
                    with gr.Row():
                        chunk_strategy = gr.Dropdown(
                            choices=["balanced", "precise", "contextual"],
                            value="balanced",
                            label="🧩 Chunking Strategy"
                        )
                        embedding_model = gr.Dropdown(
                            choices=["fast", "balanced", "multilingual"],
                            value="fast",
                            label="🧠 Embedding Model"
                        )
                    
                    enable_hierarchical = gr.Checkbox(
                        label="🔗 Enable Hierarchical Chunking",
                        value=True
                    )
                    
                    process_btn = gr.Button("🚀 Process Document", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    stats_display = gr.Markdown(rag_system._get_thread_stats_display(default_thread_id))
            
            with gr.Row():
                status_output = gr.Textbox(
                    label="Processing Status",
                    value="⏳ Awaiting document...",
                    interactive=False
                )
            
            with gr.Row():
                analysis_output = gr.Markdown("Document analysis will appear here after processing.")

        # ==================== TAB 2: CHAT INTERFACE ====================
        with gr.Tab("💬 Intelligent Chat"):
            gr.Markdown("### Chat with Documents in Current Thread")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=500
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about your documents...",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    clear_chat_btn = gr.Button("🗑️ Clear Chat Display")
                    
                    gr.Markdown("#### Example Questions")
                    with gr.Row():
                        example1 = gr.Button("What are the main findings?", size="sm")
                        example2 = gr.Button("Summarize the methodology", size="sm")
                        example3 = gr.Button("What are the key conclusions?", size="sm")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Advanced Settings")
                    
                    gr.Markdown("⚠️ **Note:** Self-RAG uses extra API calls")
                    
                    self_rag_toggle = gr.Checkbox(
                        label="🤖 Self-RAG (Query Analysis & Verification)",
                        value=False,
                        info="Uses 2 extra API calls per query"
                    )
                    
                    cache_toggle = gr.Checkbox(
                        label="💾 Semantic Caching",
                        value=True,
                        info="Saves similar queries (recommended)"
                    )
                    
                    retrieval_strategy_select = gr.Dropdown(
                        choices=["hybrid", "semantic", "keyword"],
                        value="hybrid",
                        label="🎯 Retrieval Strategy"
                    )
                    
                    temp_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.2,
                        step=0.1,
                        label="🌡️ Temperature"
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 💾 Export Options")
                    
                    export_current_btn = gr.Button("💾 Export Current Thread")
                    export_all_btn = gr.Button("📦 Export All Threads")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
        
        # ==================== TAB 3: THREAD MANAGEMENT ====================
        with gr.Tab("🧵 Thread Management"):
            gr.Markdown("### Manage Conversation Threads")
            gr.Markdown("**Key Feature:** Each thread has its own isolated PDF collection - no cross-contamination!")
            
            with gr.Row():
                with gr.Column(scale=2):
                    thread_name_input = gr.Textbox(
                        label="New Thread Name (optional)",
                        placeholder="My Research Project",
                        scale=3
                    )
                    new_thread_btn = gr.Button("➕ Create New Thread", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    global_stats_display = gr.Markdown(rag_system._get_global_stats_display())
            
            current_thread_display = gr.Textbox(
                label="Current Thread ID",
                value=default_thread_id,
                interactive=False
            )
            
            current_thread_name = gr.Textbox(
                label="Current Thread Name",
                value=rag_system.threads[default_thread_id].thread_name,
                interactive=False
            )
            
            thread_info_display = gr.Markdown(rag_system._get_thread_stats_display(default_thread_id))
            
            gr.Markdown("---")
            gr.Markdown("### All Active Threads")
            
            refresh_threads_btn = gr.Button("🔄 Refresh Thread List")
            threads_display = gr.JSON(label="Thread List", value=rag_system.list_threads())
            
            gr.Markdown("---")
            gr.Markdown("### Thread Actions")
            
            with gr.Row():
                with gr.Column():
                    thread_selector = gr.Dropdown(
                        label="Switch to Thread",
                        choices=[(t["name"], t["id"]) for t in rag_system.list_threads()],
                        interactive=True
                    )
                    switch_thread_btn = gr.Button("🔄 Switch Thread", variant="secondary")
                
                with gr.Column():
                    delete_thread_selector = gr.Dropdown(
                        label="Delete Thread",
                        choices=[(t["name"], t["id"]) for t in rag_system.list_threads() if t["id"] != default_thread_id],
                        interactive=True
                    )
                    delete_thread_btn = gr.Button("🗑️ Delete Thread", variant="stop")
            
            thread_action_status = gr.Textbox(label="Action Status", interactive=False) 
        
        
        # ==================== TAB 4: ANALYTICS ====================
        with gr.Tab("📊 Analytics & Insights"):
            gr.Markdown("### System Performance & Query Analytics for Current Thread")
            
            with gr.Row():
                refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
            
            with gr.Row():
                with gr.Column():
                    query_analytics = gr.Markdown("Query analytics will appear here")
                
                with gr.Column():
                    performance_metrics = gr.Markdown("Performance metrics will appear here")
        
        # ==================== TAB 5: HELP ====================
        with gr.Tab("ℹ️ Help & Documentation"):
            gr.Markdown("""
            ## 🎯 How to Use
            
            
            ### 2️⃣ Document Processing
            - Upload PDF, TXT, DOC, or DOCX files to current thread
            - Choose chunking strategy:
              - **Balanced**: Good for most documents (700 tokens)
              - **Precise**: Better for detailed extraction (400 tokens)
              - **Contextual**: More context per chunk (1200 tokens)
            - Select embedding model:
              - **Fast**: Quick processing, good quality
              - **Balanced**: Better semantic understanding
              - **Multilingual**: For non-English documents
            
            ### 1️⃣ Thread Management
            - Create separate threads for different research projects
            - Each thread has its own isolated PDF collection
            - Switch between threads to work on different projects
            - Delete threads when no longer needed
                        
            ### 3️⃣ Chat Features
            - **Self-RAG**: Analyzes queries and verifies answers (uses extra API calls)
            - **Semantic Caching**: Speeds up similar queries
            - **Retrieval Strategies**:
              - **Hybrid**: Balanced keyword + semantic (recommended)
              - **Semantic**: Focus on meaning and context
              - **Keyword**: Focus on exact term matching
            
            ### 4️⃣ Advanced Features
            - **Thread Isolation**: PDFs in one thread don't affect other threads
            - **Document Analysis**: Automatic structure detection
            - **Source Citations**: Detailed page and section references
            - **Query Analytics**: Track performance and confidence per thread
            - **Thread Export**: Save complete thread history with metadata
            
            ## 🔧 Technical Capabilities
            
            - ✅ Multi-thread conversation management
            - ✅ Isolated document collections per thread
            - ✅ Hierarchical chunking (parent-child relationships)
            - ✅ Hybrid retrieval (BM25 + Vector + MMR)
            - ✅ Re-ranking with FlashRank
            - ✅ Semantic caching for speed
            - ✅ Self-reflective RAG (optional)
            - ✅ Multi-document support per thread
            - ✅ Metadata extraction and filtering
            - ✅ Confidence scoring
            - ✅ Response time tracking
            
            ## 💡 Tips for Best Results
            
            1. **Organize by Project**: Create separate threads for different research topics
            2. **For Research Papers**: Use "balanced" chunking with hierarchical enabled
            3. **For Quick Facts**: Use "precise" chunking
            4. **For Complex Analysis**: Enable Self-RAG and use "hybrid" retrieval
            5. **For Speed**: Enable semantic caching and disable Self-RAG
            6. **Thread Management**: Switch threads to keep conversations organized
            
            ## 🚨 Troubleshooting
            
            - **Slow responses**: Try "fast" embedding model or disable Self-RAG
            - **Irrelevant answers**: Try "semantic" retrieval strategy
            - **Missing details**: Use "contextual" chunking
            - **API quota errors**: System auto-retries with exponential backoff
            - **Thread confusion**: Check current thread name before uploading documents
            """)
    
    # ==================== EVENT HANDLERS ====================
    
    def create_thread_handler(name):
        thread_id = rag_system.create_new_thread(name if name else None)
        thread_name = rag_system.threads[thread_id].thread_name
        threads = rag_system.list_threads()
        choices = [(t["name"], t["id"]) for t in threads]
        delete_choices = [(t["name"], t["id"]) for t in threads if t["id"] != default_thread_id]
        stats = rag_system._get_thread_stats_display(thread_id)
        global_stats = rag_system._get_global_stats_display()
        thread_info = rag_system._get_thread_stats_display(thread_id)
        return (
            thread_id, thread_id, thread_name, stats, thread_info,
            gr.update(choices=choices), threads, global_stats,
            gr.update(choices=delete_choices),
            f"✅ Created new thread: {thread_name}"
        )
    
    def switch_thread_handler(selected_thread_id):
        if selected_thread_id and selected_thread_id in rag_system.threads:
            thread_name = rag_system.threads[selected_thread_id].thread_name
            stats = rag_system._get_thread_stats_display(selected_thread_id)
            thread_info = rag_system._get_thread_stats_display(selected_thread_id)
            return (
                selected_thread_id, selected_thread_id, thread_name, stats, thread_info,
                f"✅ Switched to thread: {thread_name}", []
            )
        return (
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "⚠️ Please select a valid thread", gr.update()
        )
    
    def delete_thread_handler(thread_id_to_delete, current_tid):
        if not thread_id_to_delete:
            return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Please select a thread to delete"
        
        if thread_id_to_delete == default_thread_id:
            return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete default thread"
        
        if thread_id_to_delete == current_tid:
            return gr.update(), gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete current active thread. Switch first."
        
        if rag_system.delete_thread(thread_id_to_delete):
            threads = rag_system.list_threads()
            choices = [(t["name"], t["id"]) for t in threads]
            delete_choices = [(t["name"], t["id"]) for t in threads if t["id"] != default_thread_id]
            global_stats = rag_system._get_global_stats_display()
            return (
                threads, gr.update(choices=choices), global_stats,
                gr.update(choices=delete_choices),
                f"✅ Thread deleted successfully"
            )
        return gr.update(), gr.update(), gr.update(), gr.update(), "❌ Failed to delete thread"
    
    def refresh_threads_handler():
        threads = rag_system.list_threads()
        choices = [(t["name"], t["id"]) for t in threads]
        delete_choices = [(t["name"], t["id"]) for t in threads if t["id"] != default_thread_id]
        global_stats = rag_system._get_global_stats_display()
        return threads, gr.update(choices=choices), global_stats, gr.update(choices=delete_choices)
    
    def process_handler(file, tid, chunk_strat, embed_model, hierarchical):
        status, stats, analysis = rag_system.process_document(
            file, tid, chunk_strat, embed_model, hierarchical
        )
        tname = rag_system.threads[tid].thread_name if tid in rag_system.threads else "Unknown"
        thread_info = rag_system._get_thread_stats_display(tid)
        return status, stats, analysis, tname, thread_info
    
    def chat_fn(message, history, tid):
        response = rag_system.chat(
            message, tid, history,
            use_self_rag=self_rag_toggle.value,
            use_cache=cache_toggle.value,
            retrieval_strategy=retrieval_strategy_select.value,
            temperature=temp_slider.value
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""
    
    def clear_chat():
        return []
    
    def export_current_handler(tid):
        try:
            filename = rag_system.export_thread(tid)
            return f"✅ Thread exported to: {filename}"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def export_all_handler():
        try:
            filename = rag_system.export_all_threads()
            return f"✅ All threads exported to: {filename}"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"
    
    def refresh_analytics_handler(tid):
        analytics = rag_system.get_query_analytics(tid)
        stats = rag_system._get_thread_stats_display(tid)
        return analytics, stats
    
    # Connect handlers
    new_thread_btn.click(
        fn=create_thread_handler,
        inputs=[thread_name_input],
        outputs=[
            current_thread, current_thread_display, current_thread_name,
            stats_display, thread_info_display, thread_selector, threads_display,
            global_stats_display, delete_thread_selector, thread_action_status
        ]
    )
    
    switch_thread_btn.click(
        fn=switch_thread_handler,
        inputs=[thread_selector],
        outputs=[
            current_thread, current_thread_display, current_thread_name,
            stats_display, thread_info_display, thread_action_status, chatbot
        ]
    )
    
    delete_thread_btn.click(
        fn=delete_thread_handler,
        inputs=[delete_thread_selector, current_thread],
        outputs=[
            threads_display, thread_selector, global_stats_display,
            delete_thread_selector, thread_action_status
        ]
    )
    
    refresh_threads_btn.click(
        fn=refresh_threads_handler,
        outputs=[threads_display, thread_selector, global_stats_display, delete_thread_selector]
    )
    
    process_btn.click(
        fn=process_handler,
        inputs=[file_input, current_thread, chunk_strategy, embedding_model, enable_hierarchical],
        outputs=[status_output, stats_display, analysis_output, current_thread_name, thread_info_display]
    )
    
    submit_btn.click(
        fn=chat_fn,
        inputs=[msg_input, chatbot, current_thread],
        outputs=[chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=chat_fn,
        inputs=[msg_input, chatbot, current_thread],
        outputs=[chatbot, msg_input]
    )
    
    clear_chat_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )
    
    example1.click(lambda: "What are the main findings?", outputs=msg_input)
    example2.click(lambda: "Summarize the methodology", outputs=msg_input)
    example3.click(lambda: "What are the key conclusions?", outputs=msg_input)
    
    export_current_btn.click(
        fn=export_current_handler,
        inputs=[current_thread],
        outputs=export_status
    )
    
    export_all_btn.click(
        fn=export_all_handler,
        outputs=export_status
    )
    
    refresh_analytics_btn.click(
        fn=refresh_analytics_handler,
        inputs=[current_thread],
        outputs=[query_analytics, performance_metrics]
    )

if __name__ == "__main__":
    demo.launch(share=False)