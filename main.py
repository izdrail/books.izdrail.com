import os
import re
import uuid
import tempfile
import requests
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import hashlib
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

# Third-party imports
from num2words import num2words
import gradio as gr

# PDF Processing
import PyPDF2
import fitz  # PyMuPDF
from pdfplumber import PDF as PDFPlumber

# Audio
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
from TTS.api import TTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_tts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class Config:
    """Application configuration"""
    OLLAMA_BASE_URL: str = "http://ollama:11434"  # Updated to localhost
    DEFAULT_MODEL: str = "llama2"  # Changed to llama2
    VOICE_SAMPLES_DIR: str = "voice_samples"
    GENERATED_AUDIO_DIR: str = "generated_audio"
    PROCESSED_PDFS_DIR: str = "processed_pdfs"
    CACHE_DIR: str = "cache"
    OLLAMA_CACHE_DIR: str = "ollama_cache"  # New cache dir for Ollama results
    STANDARD_VOICE_NAME: str = "standard"
    WAV_SUFFIX: str = ".wav"
    VOICE_CONFIG_FILE: str = "voice_config.json"
    MAX_TEXT_LENGTH: int = 50000000
    MAX_CHUNK_SIZE: int = 2000
    MIN_CHUNK_SIZE: int = 200
    SUPPORTED_AUDIO_FORMATS: List[str] = None
    SUPPORTED_PDF_FORMATS: List[str] = None
    MAX_CONCURRENT_TASKS: int = 4
    OLLAMA_MAX_TOKENS: int = 4000  # Ollama context window
    OLLAMA_TEMPERATURE: float = 0.1  # Ollama temperature

    def __post_init__(self):
        if self.SUPPORTED_AUDIO_FORMATS is None:
            self.SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg']
        if self.SUPPORTED_PDF_FORMATS is None:
            self.SUPPORTED_PDF_FORMATS = ['.pdf']

        # Create directories
        for directory in [
            self.VOICE_SAMPLES_DIR,
            self.GENERATED_AUDIO_DIR,
            self.PROCESSED_PDFS_DIR,
            self.CACHE_DIR,
            self.OLLAMA_CACHE_DIR
        ]:
            Path(directory).mkdir(exist_ok=True)

config = Config()
# Enhanced Ollama Stuffing Processor with Better Error Handling
class OllamaStuffingProcessor:
    """
    Enhanced Ollama processor integrated with the PDF to TTS application
    """

    def __init__(self,
                 base_url: str = None,
                 model_name: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        """Initialize the Ollama processor with config defaults"""
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip('/')
        self.model_name = model_name or config.DEFAULT_MODEL
        self.temperature = temperature if temperature is not None else config.OLLAMA_TEMPERATURE
        self.max_tokens = max_tokens or config.OLLAMA_MAX_TOKENS
        self.api_url = f"{self.base_url}/api/generate"
        self.cache_dir = Path(config.OLLAMA_CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def check_ollama_connection(self) -> Tuple[bool, str]:
        """Check if Ollama is running and accessible with enhanced feedback"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])

                # Better error handling for empty models list
                if not models:
                    message = "⚠️ Ollama is running but no models are available. Please pull a model first."
                    logger.warning(message)
                    return False, message

                model_names = []
                for model in models:
                    if isinstance(model, dict) and 'name' in model:
                        model_names.append(model['name'])
                    else:
                        logger.warning(f"Unexpected model format: {model}")

                if not model_names:
                    message = "⚠️ No valid models found in Ollama response"
                    logger.warning(message)
                    return False, message

                logger.info(f"Connected to Ollama. Available models: {model_names}")

                # Check if requested model is available
                available = any(self.model_name in name for name in model_names)
                if not available:
                    # Try to find a close match
                    close_match = None
                    for name in model_names:
                        if self.model_name.lower() in name.lower() or name.lower() in self.model_name.lower():
                            close_match = name
                            break

                    if close_match:
                        message = f"⚠️ Exact model '{self.model_name}' not found, but found similar: '{close_match}'. Available: {', '.join(model_names)}"
                        logger.warning(message)
                        return False, message
                    else:
                        message = f"⚠️ Model '{self.model_name}' not found. Available: {', '.join(model_names)}"
                        logger.warning(message)
                        return False, message

                message = f"✅ Connected to Ollama. Using model: {self.model_name}"
                return True, message
            else:
                message = f"❌ Failed to connect to Ollama: HTTP {response.status_code}"
                logger.error(message)
                return False, message
        except requests.exceptions.ConnectionError:
            message = "❌ Cannot connect to Ollama. Is it running at " + self.base_url + "?"
            logger.error(message)
            return False, message
        except requests.exceptions.Timeout:
            message = "❌ Connection to Ollama timed out"
            logger.error(message)
            return False, message
        except Exception as e:
            message = f"❌ Connection error: {str(e)}"
            logger.error(message)
            return False, message

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count"""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _get_cache_key(self, text: str, task_type: str, custom_query: str = None) -> str:
        """Generate cache key for Ollama results"""
        content = f"{text}_{task_type}_{custom_query or ''}_{self.model_name}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load result from cache if available"""
        if not cache_key:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # Check if cache is not too old (24 hours)
                    cache_time_str = cached_data.get('cached_at', '1970-01-01')
                    try:
                        cache_time = datetime.fromisoformat(cache_time_str)
                        if (datetime.now() - cache_time).total_seconds() < 86400:
                            logger.info(f"Using cached Ollama result: {cache_key}")
                            return cached_data
                    except ValueError:
                        logger.warning(f"Invalid cache timestamp: {cache_time_str}")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """Save result to cache"""
        if not cache_key or not result:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            result['cached_at'] = datetime.now().isoformat()
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached Ollama result: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def truncate_text_if_needed(self, text: str, max_tokens: int = None) -> str:
        """Truncate text if it exceeds the token limit"""
        if not text:
            return ""

        if max_tokens is None:
            max_tokens = self.max_tokens

        estimated_tokens = self.estimate_tokens(text)

        if estimated_tokens <= max_tokens:
            return text

        # Truncate to approximately fit within token limit
        max_chars = max(100, (max_tokens - 500) * 4)  # Reserve tokens for prompt

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated due to length limitations]"
            logger.warning(f"Text truncated: {estimated_tokens} → ~{self.estimate_tokens(text)} tokens")

        return text

    def create_stuffing_prompt(self,
                             document_text: str,
                             task_type: str = "summarize",
                             custom_query: str = None) -> str:
        """Create enhanced prompts for different processing tasks"""

        if not document_text or not document_text.strip():
            raise ValueError("Document text is empty")

        # Ensure text fits within token limits
        document_text = self.truncate_text_if_needed(document_text)

        prompts = {
            "summarize": f"""Please provide a comprehensive and well-structured summary of the following document. Focus on:
- Main topics and key themes
- Important facts, findings, or conclusions
- Critical data points or statistics
- Actionable insights or recommendations

Make the summary clear, concise, and suitable for text-to-speech conversion. Use proper sentence structure and avoid special characters.

Document:
{document_text}

Summary:""",

            "analyze": f"""Analyze the following document in detail and provide insights about:

1. **Main Themes**: What are the central topics and ideas?
2. **Key Arguments**: What are the main points being made?
3. **Evidence & Data**: What supporting information is provided?
4. **Conclusions**: What are the final recommendations or findings?
5. **Implications**: What are the broader meanings or consequences?

Present your analysis in clear, flowing prose suitable for audio conversion.

Document:
{document_text}

Analysis:""",

            "extract": f"""Extract and organize the most important information from this document:

**Key Facts & Figures:**
[List important data points, statistics, dates, names]

**Main Conclusions:**
[Summarize primary findings or recommendations]

**Important Quotes:**
[Notable statements or key phrases]

**Critical Details:**
[Essential information that shouldn't be missed]

Make the extraction clear and well-organized for audio presentation.

Document:
{document_text}

Extracted Information:""",

            "simplify": f"""Rewrite the following document in simpler, clearer language suitable for a general audience. Make it:
- Easy to understand and follow
- Well-structured with smooth transitions
- Free of jargon and technical complexity
- Perfect for text-to-speech conversion

Maintain all important information while making it more accessible.

Original Document:
{document_text}

Simplified Version:""",

            "outline": f"""Create a detailed outline of the following document with:
- Main sections and subsections
- Key points under each section
- Important supporting details
- Logical flow and structure

Format as a clear, hierarchical outline suitable for audio presentation.

Document:
{document_text}

Detailed Outline:""",

            "custom": f"""Based on the following document, please answer this specific question: {custom_query}

Provide a thorough, well-structured response using only information from the document. If the answer requires inference, clearly indicate that. If information is not available in the document, state that explicitly.

Document:
{document_text}

Question: {custom_query}

Answer:"""
        }

        if task_type not in prompts:
            if task_type == "custom" and not custom_query:
                raise ValueError("Custom query required for 'custom' task type")
            else:
                available_tasks = ", ".join(prompts.keys())
                raise ValueError(f"Invalid task_type '{task_type}'. Available: {available_tasks}")

        return prompts[task_type]

    def send_to_ollama(self, prompt: str, progress_callback=None) -> Dict[str, any]:
        """Send request to Ollama with enhanced error handling"""
        if not prompt or not prompt.strip():
            return {"success": False, "error": "Empty prompt provided"}

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 2048,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        try:
            logger.info(f"Sending request to Ollama (model: {self.model_name})...")
            start_time = time.time()

            if progress_callback:
                progress_callback(0.1)  # Starting request

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300,
                headers={'Content-Type': 'application/json'}
            )

            if progress_callback:
                progress_callback(0.9)  # Processing complete

            end_time = time.time()
            processing_time = end_time - start_time

            if response.status_code == 200:
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Ollama response as JSON: {e}")
                    return {
                        "success": False,
                        "error": f"Invalid JSON response: {e}",
                        "processing_time": processing_time
                    }

                # Extract response text with better error handling
                response_text = result.get("response", "")
                if not response_text:
                    logger.warning("Empty response from Ollama")
                    return {
                        "success": False,
                        "error": "Empty response from Ollama",
                        "processing_time": processing_time
                    }

                if progress_callback:
                    progress_callback(1.0)  # Finished

                logger.info(f"Ollama request completed in {processing_time:.2f} seconds")
                return {
                    "success": True,
                    "response": response_text,
                    "processing_time": processing_time,
                    "model": result.get("model", self.model_name),
                    "prompt_tokens": self.estimate_tokens(prompt),
                    "response_tokens": self.estimate_tokens(response_text),
                    "context": result.get("context", [])
                }
            else:
                error_text = response.text if response.text else f"HTTP {response.status_code}"
                logger.error(f"Ollama API error: {error_text}")
                return {
                    "success": False,
                    "error": error_text,
                    "processing_time": processing_time
                }

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return {"success": False, "error": "Request timed out after 5 minutes"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error in send_to_ollama: {e}")
            return {"success": False, "error": f"Unexpected error: {e}"}

    def process_document(self,
                        cleaned_text: str,
                        task_type: str = "summarize",
                        custom_query: str = None,
                        use_cache: bool = True,
                        progress_callback=None) -> Dict[str, any]:
        """Process document with enhanced error handling and validation"""

        try:
            # Validate inputs
            if not cleaned_text or not cleaned_text.strip():
                return {"success": False, "error": "No text provided for processing"}

            if not task_type:
                return {"success": False, "error": "No task type specified"}

            # Check connection first
            is_connected, connection_msg = self.check_ollama_connection()
            if not is_connected:
                return {"success": False, "error": connection_msg}

            # Check cache if enabled
            if use_cache:
                try:
                    cache_key = self._get_cache_key(cleaned_text, task_type, custom_query)
                    cached_result = self._load_from_cache(cache_key)
                    if cached_result:
                        if progress_callback:
                            progress_callback(1.0)
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache check failed: {e}")

            # Create the prompt
            try:
                prompt = self.create_stuffing_prompt(cleaned_text, task_type, custom_query)
                logger.info(f"Created prompt for task: {task_type} ({len(prompt):,} chars)")
            except Exception as e:
                logger.error(f"Error creating prompt: {e}")
                return {"success": False, "error": f"Prompt creation failed: {e}"}

            # Send to Ollama
            result = self.send_to_ollama(prompt, progress_callback)

            # Cache successful results
            if result.get("success") and use_cache:
                try:
                    cache_key = self._get_cache_key(cleaned_text, task_type, custom_query)
                    self._save_to_cache(cache_key, result)
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")

            return result

        except Exception as e:
            logger.error(f"Unexpected error in process_document: {e}")
            return {"success": False, "error": f"Processing failed: {e}"}

# Enhanced function to get available models with better error handling
def get_available_ollama_models():
    """Get list of available Ollama models with enhanced error handling"""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get('models', [])

            if not models:
                logger.warning("No models available in Ollama")
                return [config.DEFAULT_MODEL]

            model_names = []
            for model in models:
                if isinstance(model, dict) and 'name' in model:
                    model_names.append(model['name'])
                else:
                    logger.warning(f"Unexpected model format: {model}")

            return model_names if model_names else [config.DEFAULT_MODEL]
        else:
            logger.error(f"Failed to get models: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")

    return [config.DEFAULT_MODEL]

# Enhanced process_with_ollama function with better error handling
def process_with_ollama(text: str, task_type: str, custom_query: str = None,
                       use_cache: bool = True, progress=None):
    """Process text with Ollama using the stuffing method - Enhanced version"""
    try:
        # Input validation
        if not text or not text.strip():
            return "", "❌ No text provided for Ollama processing!"

        if not task_type:
            return "", "❌ Please select a processing task!"

        # Progress callback for Gradio
        def progress_callback(ratio):
            if progress and hasattr(progress, '__call__'):
                try:
                    progress(ratio)
                except:
                    pass

        logger.info(f"Starting Ollama processing: {task_type}")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Custom query: {custom_query if custom_query else 'None'}")

        # Process with Ollama
        result = ollama_processor.process_document(
            text,
            task_type,
            custom_query,
            use_cache,
            progress_callback
        )

        if result.get("success"):
            response_text = result.get("response", "")
            if not response_text:
                return "", "❌ Ollama returned empty response!"

            processing_time = result.get("processing_time", 0)
            model_used = result.get("model", "unknown")
            prompt_tokens = result.get("prompt_tokens", 0)
            response_tokens = result.get("response_tokens", 0)

            status_msg = f"""✅ Ollama processing completed successfully!
🤖 Model: {model_used}
⏱️ Time: {processing_time:.2f} seconds
📊 Input tokens: ~{prompt_tokens:,}
📝 Output tokens: ~{response_tokens:,}
🎯 Task: {task_type}
💾 Cache used: {use_cache}
📄 Output length: {len(response_text):,} characters"""

            if custom_query:
                status_msg += f"\n❓ Query: {custom_query[:100]}..."

            # Save result for reference
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"ollama_{task_type}_{timestamp}.txt"
                result_path = Path(config.PROCESSED_PDFS_DIR) / result_filename

                metadata = {
                    "task_type": task_type,
                    "custom_query": custom_query,
                    "model": model_used,
                    "processing_time": processing_time,
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "processed_at": datetime.now().isoformat(),
                    "input_length": len(text),
                    "output_length": len(response_text)
                }

                with open(result_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Ollama Processing Result\n")
                    f.write(f"# Metadata: {json.dumps(metadata, indent=2)}\n\n")
                    f.write(f"# Original Task: {task_type}\n")
                    if custom_query:
                        f.write(f"# Custom Query: {custom_query}\n")
                    f.write(f"\n# Processed Text:\n{response_text}")

                status_msg += f"\n💾 Result saved: {result_filename}"
            except Exception as e:
                logger.warning(f"Failed to save result: {e}")
                status_msg += f"\n⚠️ Could not save result file: {e}"

            return response_text, status_msg

        else:
            error_msg = result.get("error", "Unknown error occurred")
            logger.error(f"Ollama processing failed: {error_msg}")
            return "", f"❌ Ollama processing failed: {error_msg}"

    except Exception as e:
        logger.error(f"Ollama processing error: {e}", exc_info=True)
        return "", f"❌ Processing error: {str(e)}"

# --- Existing classes remain the same ---
class FileUtils:
    """Utility functions for file operations"""

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate MD5 hash of file for caching"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash for {file_path}: {e}")
            return str(uuid.uuid4())

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 100) -> str:
        """Sanitize filename for safe file operations"""
        sanitized = re.sub(r'[^\w\s\-\.]', '', filename)
        sanitized = re.sub(r'[\s]+', '_', sanitized)
        return sanitized[:max_length]

    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0.0

# --- Enhanced PDF Text Extractor ---
class PDFTextExtractor:
    """Enhanced PDF text extraction with caching and better error handling"""

    def __init__(self):
        self.extraction_methods = ['pdfplumber', 'pymupdf', 'pypdf2']
        self.cache_dir = Path(config.CACHE_DIR) / "pdf_extractions"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, pdf_path: str, method: str) -> str:
        """Get cache file path for extracted text"""
        file_hash = FileUtils.get_file_hash(pdf_path)
        cache_filename = f"{file_hash}_{method}.txt"
        return str(self.cache_dir / cache_filename)

    def _load_from_cache(self, pdf_path: str, method: str) -> Optional[str]:
        """Load extracted text from cache if available"""
        cache_path = self._get_cache_path(pdf_path, method)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Loaded text from cache: {cache_path}")
                    return f.read()
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")
        return None

    def _save_to_cache(self, pdf_path: str, method: str, text: str):
        """Save extracted text to cache"""
        cache_path = self._get_cache_path(pdf_path, method)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved text to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (best for complex layouts)"""
        text = ""
        try:
            with PDFPlumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Extracting {total_pages} pages with pdfplumber")

                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"

                        if i % 10 == 0 and i > 0:
                            logger.info(f"Processed {i}/{total_pages} pages")
                    except Exception as e:
                        logger.warning(f"Error extracting page {i}: {e}")
                        continue

        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            raise
        return text

    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (good balance of speed and accuracy)"""
        text = ""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            logger.info(f"Extracting {total_pages} pages with PyMuPDF")

            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n\n"

                    if page_num % 10 == 0 and page_num > 0:
                        logger.info(f"Processed {page_num}/{total_pages} pages")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            raise
        finally:
            if doc:
                doc.close()
        return text

    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Extracting {total_pages} pages with PyPDF2")

                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"

                        if i % 10 == 0 and i > 0:
                            logger.info(f"Processed {i}/{total_pages} pages")
                    except Exception as e:
                        logger.warning(f"Error extracting page {i}: {e}")
                        continue

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
        return text

    def validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """Validate PDF file before processing"""
        if not os.path.exists(pdf_path):
            return False, f"File not found: {pdf_path}"

        file_size = FileUtils.get_file_size_mb(pdf_path)
        if file_size > 100:
            return False, f"File too large: {file_size:.1f}MB (max 100MB)"

        if file_size == 0:
            return False, "File is empty"

        try:
            with open(pdf_path, 'rb') as f:
                PyPDF2.PdfReader(f)
            return True, "Valid PDF"
        except Exception as e:
            return False, f"Invalid PDF file: {e}"

    def extract_text(self, pdf_path: str, method: str = 'auto') -> str:
        """Extract text from PDF with enhanced error handling and caching"""
        is_valid, message = self.validate_pdf(pdf_path)
        if not is_valid:
            raise ValueError(message)

        logger.info(f"Starting text extraction from {pdf_path} using method: {method}")

        if method == 'auto':
            for extraction_method in self.extraction_methods:
                cached_text = self._load_from_cache(pdf_path, extraction_method)
                if cached_text:
                    return cached_text

                try:
                    text = self._extract_by_method(pdf_path, extraction_method)
                    if text.strip():
                        logger.info(f"Successfully extracted text using {extraction_method}")
                        self._save_to_cache(pdf_path, extraction_method, text)
                        return text
                except Exception as e:
                    logger.warning(f"{extraction_method} failed: {e}")
                    continue

            raise Exception("All extraction methods failed")
        else:
            cached_text = self._load_from_cache(pdf_path, method)
            if cached_text:
                return cached_text

            text = self._extract_by_method(pdf_path, method)
            if text.strip():
                self._save_to_cache(pdf_path, method, text)
            return text

    def _extract_by_method(self, pdf_path: str, method: str) -> str:
        """Extract text using specific method"""
        method_map = {
            'pdfplumber': self.extract_with_pdfplumber,
            'pymupdf': self.extract_with_pymupdf,
            'pypdf2': self.extract_with_pypdf2
        }

        if method not in method_map:
            raise ValueError(f"Unknown extraction method: {method}")

        return method_map[method](pdf_path)

# --- Enhanced Text Cleaner ---
class TextCleaner:
    """Enhanced text cleaning with better preprocessing and validation"""

    def __init__(self):
        self.patterns = {
            'headers_footers': r'^(Page \d+|\d+\s*$|Chapter \d+.*$)',
            'figure_refs': r'(Figure|Fig\.|Table|Tbl\.)\s*\d+[\w\s]*',
            'citations': r'\[\d+\]|\(\w+,?\s*\d{4}\)|\(\w+\s+et\s+al\.,?\s*\d{4}\)',
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'whitespace': r'\s{3,}',
            'broken_sentences': r'(?<=[a-z])\n(?=[a-z])',
        }

        self.abbreviations = {
            'Dr.': 'Doctor', 'Prof.': 'Professor', 'vs.': 'versus',
            'etc.': 'et cetera', 'i.e.': 'that is', 'e.g.': 'for example',
            'Mr.': 'Mister', 'Mrs.': 'Misses', 'Ms.': 'Miss',
            'Inc.': 'Incorporated', 'Corp.': 'Corporation', 'Ltd.': 'Limited',
            'Ave.': 'Avenue', 'St.': 'Street', 'Rd.': 'Road',
            'Jan.': 'January', 'Feb.': 'February', 'Mar.': 'March',
            'Apr.': 'April', 'Jun.': 'June', 'Jul.': 'July',
            'Aug.': 'August', 'Sep.': 'September', 'Oct.': 'October',
            'Nov.': 'November', 'Dec.': 'December'
        }

    def validate_text(self, text: str) -> Tuple[bool, str]:
        """Validate text before processing"""
        if not text or not isinstance(text, str):
            return False, "Text is empty or invalid"

        if len(text.strip()) == 0:
            return False, "Text contains only whitespace"

        if len(text) > config.MAX_TEXT_LENGTH:
            return False, f"Text too long: {len(text)} characters (max {config.MAX_TEXT_LENGTH})"

        printable_chars = sum(1 for c in text if c.isprintable())
        if printable_chars / len(text) < 0.8:
            return False, "Text contains too many non-printable characters"

        return True, "Valid text"

    def clean_academic_text(self, text: str) -> str:
        """Clean academic/research paper text"""
        patterns_to_remove = [
            r'Abstract\s*\n',
            r'References\s*\n.*$',
            r'Bibliography\s*\n.*$',
            r'\b(doi|DOI):\s*[\w\./\-]+',
            r'Acknowledgments?\s*\n.*?(?=\n\n|\n[A-Z]|\Z)',
            r'Keywords?:\s*.*?\n',
            r'©\s*\d{4}.*?\n'
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        return text

    def expand_numbers(self, text: str) -> str:
        """Convert numbers to words for better TTS"""
        def replace_number(match):
            try:
                num = int(match.group(0))
                if 0 <= num <= 9999:
                    return num2words(num)
                return match.group(0)
            except:
                return match.group(0)

        text = re.sub(r'\b(?<![\d.])\d+(?![\d.])\b', replace_number, text)
        return text

    def clean_for_tts(self, text: str, preserve_structure: bool = True) -> str:
        """Enhanced text cleaning for TTS conversion"""
        is_valid, message = self.validate_text(text)
        if not is_valid:
            raise ValueError(f"Text validation failed: {message}")

        logger.info(f"Cleaning text for TTS (length: {len(text)} chars)")
        original_length = len(text)

        text = text.strip()

        # Remove or replace problematic characters for TTS
        text = re.sub(r'[^\w\s.,!?;:\-\'"()\n\r]', ' ', text)

        # Fix broken sentences (common in PDFs)
        text = re.sub(self.patterns['broken_sentences'], ' ', text)

        # Remove excessive whitespace
        text = re.sub(self.patterns['whitespace'], ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove citations and references
        text = re.sub(self.patterns['citations'], '', text)
        text = re.sub(self.patterns['figure_refs'], '', text)

        # Remove URLs and emails
        text = re.sub(self.patterns['urls'], '', text)
        text = re.sub(self.patterns['emails'], '', text)

        # Expand abbreviations
        for abbrev, expansion in self.abbreviations.items():
            text = text.replace(abbrev, expansion)

        # Convert numbers to words
        text = self.expand_numbers(text)

        # Ensure proper sentence structure
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)

        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()

        # Preserve paragraph structure if requested
        if preserve_structure:
            text = re.sub(r'\n\s*\n', '\n\n', text)
        else:
            text = re.sub(r'\n+', ' ', text)

        cleaned_length = len(text)
        logger.info(f"Text cleaned: {original_length} -> {cleaned_length} characters")

        return text

    def split_into_chunks(self, text: str, max_chars: int = 1000) -> List[str]:
        """Enhanced text chunking with better sentence boundary detection"""
        max_chars = max(config.MIN_CHUNK_SIZE, min(max_chars, config.MAX_CHUNK_SIZE))

        if len(text) <= max_chars:
            return [text]

        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)

                for sentence in sentences:
                    if len(current_chunk + sentence) <= max_chars:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                if len(current_chunk + paragraph) <= max_chars:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= config.MIN_CHUNK_SIZE]

        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks

# --- Enhanced Voice Management ---
class VoiceManager:
    """Enhanced voice management with better validation and metadata"""

    def __init__(self):
        self.voice_config_path = config.VOICE_CONFIG_FILE
        self.voices = self.load_voice_config()
        self._lock = threading.RLock()

    def load_voice_config(self) -> Dict:
        """Load voice configuration from JSON file with error handling"""
        if os.path.exists(self.voice_config_path):
            try:
                with open(self.voice_config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                logger.info("Voice configuration loaded successfully")
                return config_data
            except Exception as e:
                logger.error(f"Error loading voice config: {e}")

        default_config = {
            "standard": {
                "name": "Standard Voice",
                "description": "Default synthetic voice",
                "type": "standard",
                "enabled": True,
                "quality": "medium",
                "created_at": datetime.now().isoformat()
            }
        }
        self.save_voice_config(default_config)
        return default_config

    def save_voice_config(self, config_data: Dict):
        """Save voice configuration to JSON file with backup"""
        with self._lock:
            try:
                if os.path.exists(self.voice_config_path):
                    backup_path = f"{self.voice_config_path}.backup"
                    os.rename(self.voice_config_path, backup_path)

                with open(self.voice_config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                logger.info("Voice configuration saved successfully")

                backup_path = f"{self.voice_config_path}.backup"
                if os.path.exists(backup_path):
                    os.remove(backup_path)

            except Exception as e:
                logger.error(f"Error saving voice config: {e}")
                backup_path = f"{self.voice_config_path}.backup"
                if os.path.exists(backup_path):
                    os.rename(backup_path, self.voice_config_path)
                raise

    def validate_voice_sample(self, audio_path: str) -> Tuple[bool, str]:
        """Validate uploaded voice sample"""
        if not os.path.exists(audio_path):
            return False, "Audio file not found"

        file_size = FileUtils.get_file_size_mb(audio_path)
        if file_size > 50:
            return False, f"Audio file too large: {file_size:.1f}MB (max 50MB)"

        if file_size < 0.1:
            return False, "Audio file too small (min 100KB)"

        file_ext = Path(audio_path).suffix.lower()
        if file_ext not in config.SUPPORTED_AUDIO_FORMATS:
            return False, f"Unsupported format: {file_ext}"

        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate

            if duration < 5:
                return False, f"Audio too short: {duration:.1f}s (min 5s)"

            if duration > 300:
                return False, f"Audio too long: {duration:.1f}s (max 300s)"

            return True, f"Valid audio: {duration:.1f}s at {sample_rate}Hz"

        except Exception as e:
            return False, f"Invalid audio file: {e}"

    def get_available_voices(self) -> List[str]:
        """Get list of available and enabled voice IDs"""
        with self._lock:
            voices = []

            if self.voices.get("standard", {}).get("enabled", True):
                voices.append("standard")

            if os.path.exists(config.VOICE_SAMPLES_DIR):
                for voice_dir in os.listdir(config.VOICE_SAMPLES_DIR):
                    voice_path = os.path.join(config.VOICE_SAMPLES_DIR, voice_dir)
                    reference_file = os.path.join(voice_path, "reference.wav")

                    if os.path.isdir(voice_path) and os.path.exists(reference_file):
                        voice_info = self.voices.get(voice_dir, {})
                        if voice_info.get("enabled", True):
                            voices.append(voice_dir)

            return voices if voices else ["standard"]

    def get_voice_display_names(self) -> List[str]:
        """Get list of voice display names with metadata"""
        voices = self.get_available_voices()
        display_names = []

        for voice_id in voices:
            if voice_id == "standard":
                display_names.append("🔊 Standard Voice")
            else:
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                quality = voice_info.get("quality", "unknown")
                display_names.append(f"🎭 {name} ({quality})")

        return display_names

    def get_voice_id_from_display(self, display_name: str) -> str:
        """Get voice ID from display name"""
        if "Standard Voice" in display_name:
            return "standard"

        for voice_id in self.get_available_voices():
            if voice_id != "standard":
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                quality = voice_info.get("quality", "unknown")
                if display_name == f"🎭 {name} ({quality})":
                    return voice_id

        return "standard"

    def add_voice(self, voice_id: str, name: str, description: str = "",
                  quality: str = "medium") -> bool:
        """Add a new voice to the configuration with validation"""
        with self._lock:
            try:
                self.voices[voice_id] = {
                    "name": name,
                    "description": description,
                    "type": "cloned",
                    "quality": quality,
                    "enabled": True,
                    "created_at": datetime.now().isoformat(),
                    "file_hash": FileUtils.get_file_hash(
                        os.path.join(config.VOICE_SAMPLES_DIR, voice_id, "reference.wav")
                    )
                }
                self.save_voice_config(self.voices)
                logger.info(f"Voice '{name}' added successfully")
                return True
            except Exception as e:
                logger.error(f"Error adding voice: {e}")
                return False

# --- Enhanced TTS Generator ---
class TTSGenerator:
    """Enhanced TTS generator with better performance and error handling"""

    def __init__(self):
        self.voice_model = None
        self.tacotron2 = None
        self.hifi_gan = None
        self.voice_manager = VoiceManager()
        self.model_lock = threading.RLock()
        self._model_loaded = {"standard": False, "cloned": False}

    @lru_cache(maxsize=32)
    def preprocess_text(self, text: str) -> str:
        """Cached text preprocessing"""
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def _generate_filename(self, text: str, speaker_id: str,
                          chunk_idx: Optional[int] = None,
                          fmt: str = "wav") -> str:
        """Generate safe filename for audio output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = FileUtils.sanitize_filename(text[:30]) or "audio"
        safe_speaker = FileUtils.sanitize_filename(speaker_id)

        chunk_suffix = f"_part{chunk_idx:03d}" if chunk_idx is not None else ""
        filename = f"{timestamp}_{safe_speaker}_{safe_text}{chunk_suffix}.{fmt}"

        return os.path.join(config.GENERATED_AUDIO_DIR, filename)

    def _load_standard_models(self):
        """Load standard TTS models with error handling"""
        with self.model_lock:
            if self._model_loaded["standard"]:
                return

            try:
                logger.info("Loading standard TTS models...")
                self.tacotron2 = Tacotron2.from_hparams(
                    source="speechbrain/tts-tacotron2-ljspeech",
                    savedir="tmp_tts"
                )
                self.hifi_gan = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-ljspeech",
                    savedir="tmp_vocoder"
                )
                self._model_loaded["standard"] = True
                logger.info("Standard TTS models loaded successfully")
            except Exception as e:
                logger.error(f"Error loading standard TTS models: {e}")
                raise

    def _load_cloning_model(self):
        """Load voice cloning model with error handling"""
        with self.model_lock:
            if self._model_loaded["cloned"]:
                return

            try:
                logger.info("Loading voice cloning model...")
                self.voice_model = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False
                )
                self._model_loaded["cloned"] = True
                logger.info("Voice cloning model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading voice cloning model: {e}")
                raise

    def run_standard_tts(self, text: str, chunk_idx: Optional[int] = None) -> str:
        """Generate audio using standard TTS with enhanced error handling"""
        if not text.strip():
            raise ValueError("Empty text provided for TTS")

        self._load_standard_models()

        try:
            processed_text = self.preprocess_text(text)
            logger.info(f"Generating standard TTS for text: {processed_text[:50]}...")

            mel_outputs, _, _ = self.tacotron2.encode_batch([processed_text])
            waveform = self.hifi_gan.decode_batch(mel_outputs).squeeze().detach().cpu()

            file_path = self._generate_filename(text, config.STANDARD_VOICE_NAME, chunk_idx)
            torchaudio.save(file_path, waveform.unsqueeze(0), 22050)

            logger.info(f"Standard TTS audio saved: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Standard TTS generation failed: {e}")
            raise

    def run_voice_clone_tts(self, text: str, speaker_id: str,
                           chunk_idx: Optional[int] = None) -> str:
        """Generate audio using voice cloning with enhanced error handling"""
        if not text.strip():
            raise ValueError("Empty text provided for TTS")

        self._load_cloning_model()

        reference_audio = os.path.join(config.VOICE_SAMPLES_DIR, speaker_id, "reference.wav")
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio for '{speaker_id}' not found at {reference_audio}")

        try:
            processed_text = self.preprocess_text(text)
            logger.info(f"Generating cloned TTS for speaker '{speaker_id}': {processed_text[:50]}...")

            file_path = self._generate_filename(text, speaker_id, chunk_idx)

            self.voice_model.tts_to_file(
                text=processed_text,
                file_path=file_path,
                speaker_wav=reference_audio,
                language="en",
                temperature=0.7,
                length_penalty=1.0,
                repetition_penalty=5.0,
                top_k=50,
                top_p=0.85,
                speed=1.0,
                enable_text_splitting=True
            )

            logger.info(f"Cloned TTS audio saved: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Voice cloning TTS generation failed: {e}")
            raise

    def generate_audio_from_chunks(self, text_chunks: List[str], voice_id: str,
                                 progress_callback=None) -> List[str]:
        """Generate audio from text chunks with parallel processing"""
        if not text_chunks:
            raise ValueError("No text chunks provided")

        audio_files = []
        failed_chunks = []

        def generate_chunk_audio(chunk_data):
            """Process a single chunk"""
            chunk_idx, chunk_text = chunk_data
            if not chunk_text.strip():
                return None

            try:
                if voice_id == "standard":
                    return self.run_standard_tts(chunk_text, chunk_idx + 1)
                else:
                    return self.run_voice_clone_tts(chunk_text, voice_id, chunk_idx + 1)
            except Exception as e:
                logger.error(f"Failed to generate audio for chunk {chunk_idx + 1}: {e}")
                failed_chunks.append(chunk_idx + 1)
                return None

        with ThreadPoolExecutor(max_workers=min(config.MAX_CONCURRENT_TASKS, len(text_chunks))) as executor:
            future_to_chunk = {
                executor.submit(generate_chunk_audio, (i, chunk)): i
                for i, chunk in enumerate(text_chunks)
            }

            completed = 0
            for future in as_completed(future_to_chunk):
                result = future.result()
                if result:
                    audio_files.append(result)

                completed += 1
                if progress_callback:
                    progress_callback(completed / len(text_chunks))

        audio_files.sort(key=lambda x: int(re.search(r'_part(\d+)', x).group(1)) if re.search(r'_part(\d+)', x) else 0)

        if failed_chunks:
            logger.warning(f"Failed to generate audio for {len(failed_chunks)} chunks: {failed_chunks}")

        logger.info(f"Successfully generated {len(audio_files)} audio files from {len(text_chunks)} chunks")
        return audio_files

    def generate_audio(self, text: str, voice_id: str) -> str:
        """Generate single audio file with enhanced error handling"""
        if not text.strip():
            raise ValueError("Empty text provided for TTS")

        if len(text) > 5000:
            logger.warning(f"Text is quite long ({len(text)} chars) for single file generation")

        try:
            if voice_id == "standard":
                return self.run_standard_tts(text)
            else:
                return self.run_voice_clone_tts(text, voice_id)
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old generated audio files"""
        try:
            current_time = datetime.now()
            deleted_count = 0

            for audio_file in Path(config.GENERATED_AUDIO_DIR).glob("*.wav"):
                file_age = current_time - datetime.fromtimestamp(audio_file.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    audio_file.unlink()
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old audio files")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# --- Global instances ---
pdf_extractor = PDFTextExtractor()
text_cleaner = TextCleaner()
voice_manager = VoiceManager()
tts_generator = TTSGenerator()
ollama_processor = OllamaStuffingProcessor()  # New global instance

# --- Enhanced Text Editor Functions ---
class TextEditor:
    """Enhanced text editing utilities"""

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        if not text:
            return 0
        return len(text.split())

    @staticmethod
    def count_characters(text: str) -> int:
        """Count characters in text"""
        return len(text) if text else 0

    @staticmethod
    def find_text_matches(text: str, find_text: str) -> int:
        """Find number of matches for search text"""
        if not text or not find_text:
            return 0
        return len(re.findall(re.escape(find_text), text, re.IGNORECASE))

    @staticmethod
    def replace_text(text: str, find_text: str, replace_text: str) -> Tuple[str, int]:
        """Replace all occurrences and return new text and count"""
        if not text or not find_text:
            return text, 0

        original_count = TextEditor.find_text_matches(text, find_text)
        if original_count == 0:
            return text, 0

        new_text = re.sub(re.escape(find_text), replace_text, text, flags=re.IGNORECASE)
        return new_text, original_count

    @staticmethod
    def clean_formatting(text: str) -> str:
        """Clean common formatting issues"""
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

        return text.strip()

    @staticmethod
    def estimate_reading_time(text: str, wpm: int = 200) -> str:
        """Estimate reading time in minutes"""
        if not text:
            return "0 minutes"

        word_count = TextEditor.count_words(text)
        minutes = max(1, round(word_count / wpm))

        if minutes == 1:
            return "1 minute"
        elif minutes < 60:
            return f"{minutes} minutes"
        else:
            hours = minutes // 60
            remaining_minutes = minutes % 60
            if hours == 1 and remaining_minutes == 0:
                return "1 hour"
            elif remaining_minutes == 0:
                return f"{hours} hours"
            else:
                return f"{hours}h {remaining_minutes}m"

# --- Enhanced Ollama Processing Functions ---
def check_ollama_status():
    """Check Ollama connection status"""
    try:
        is_connected, message = ollama_processor.check_ollama_connection()
        return message
    except Exception as e:
        return f"❌ Error checking Ollama: {str(e)}"

def process_with_ollama(text: str, task_type: str, custom_query: str = None,
                       use_cache: bool = True, progress=gr.Progress()):
    """Process text with Ollama using the stuffing method"""
    if not text or not text.strip():
        return "", "❌ No text provided for Ollama processing!"

    if not task_type:
        return "", "❌ Please select a processing task!"

    try:
        # Progress callback for Gradio
        def progress_callback(ratio):
            if progress:
                progress(ratio)

        logger.info(f"Starting Ollama processing: {task_type}")

        # Process with Ollama
        result = ollama_processor.process_document(
            text,
            task_type,
            custom_query,
            use_cache,
            progress_callback
        )

        if result["success"]:
            response_text = result["response"]
            processing_time = result.get("processing_time", 0)
            model_used = result.get("model", "unknown")
            prompt_tokens = result.get("prompt_tokens", 0)
            response_tokens = result.get("response_tokens", 0)

            status_msg = f"""✅ Ollama processing completed successfully!
🤖 Model: {model_used}
⏱️ Time: {processing_time:.2f} seconds
📊 Input tokens: ~{prompt_tokens:,}
📝 Output tokens: ~{response_tokens:,}
🎯 Task: {task_type}
💾 Cache used: {use_cache}
📄 Output length: {len(response_text):,} characters"""

            if custom_query:
                status_msg += f"\n❓ Query: {custom_query[:100]}..."

            # Save result for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"ollama_{task_type}_{timestamp}.txt"
            result_path = Path(config.PROCESSED_PDFS_DIR) / result_filename

            metadata = {
                "task_type": task_type,
                "custom_query": custom_query,
                "model": model_used,
                "processing_time": processing_time,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "processed_at": datetime.now().isoformat(),
                "input_length": len(text),
                "output_length": len(response_text)
            }

            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"# Ollama Processing Result\n")
                f.write(f"# Metadata: {json.dumps(metadata, indent=2)}\n\n")
                f.write(f"# Original Task: {task_type}\n")
                if custom_query:
                    f.write(f"# Custom Query: {custom_query}\n")
                f.write(f"\n# Processed Text:\n{response_text}")

            status_msg += f"\n💾 Result saved: {result_filename}"

            return response_text, status_msg

        else:
            error_msg = result.get("error", "Unknown error")
            return "", f"❌ Ollama processing failed: {error_msg}"

    except Exception as e:
        logger.error(f"Ollama processing error: {e}")
        return "", f"❌ Processing error: {str(e)}"

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            return model_names if model_names else [config.DEFAULT_MODEL]
    except:
        pass
    return [config.DEFAULT_MODEL]

def update_ollama_model(model_name: str):
    """Update the Ollama model being used"""
    if model_name:
        ollama_processor.model_name = model_name
        return f"✅ Ollama model updated to: {model_name}"
    return "❌ No model selected"

# --- Enhanced Gradio UI Functions ---
def process_pdf_enhanced(pdf_file, extraction_method, preserve_structure, chunk_size):
    """Enhanced PDF processing with better validation and progress tracking"""
    if not pdf_file:
        return "", "", "❌ Please upload a PDF file!", gr.update(visible=False)

    try:
        logger.info(f"Starting PDF processing: {pdf_file.name}")

        is_valid, validation_message = pdf_extractor.validate_pdf(pdf_file.name)
        if not is_valid:
            return "", "", f"❌ {validation_message}", gr.update(visible=False)

        raw_text = pdf_extractor.extract_text(pdf_file.name, extraction_method)

        if not raw_text.strip():
            return "", "", "❌ No text could be extracted from the PDF!", gr.update(visible=False)

        is_valid, validation_message = text_cleaner.validate_text(raw_text)
        if not is_valid:
            return "", "", f"❌ Text validation failed: {validation_message}", gr.update(visible=False)

        cleaned_text = text_cleaner.clean_for_tts(raw_text, preserve_structure)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = FileUtils.sanitize_filename(os.path.splitext(os.path.basename(pdf_file.name))[0])
        output_path = Path(config.PROCESSED_PDFS_DIR) / f"{timestamp}_{filename}_cleaned.txt"

        metadata = {
            "original_file": pdf_file.name,
            "extraction_method": extraction_method,
            "processed_at": datetime.now().isoformat(),
            "original_length": len(raw_text),
            "cleaned_length": len(cleaned_text),
            "preserve_structure": preserve_structure
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Metadata\n{json.dumps(metadata, indent=2)}\n\n# Cleaned Text\n{cleaned_text}")

        chunks = text_cleaner.split_into_chunks(cleaned_text, chunk_size)

        status = f"""✅ PDF processed successfully!
📄 Original: {len(raw_text):,} characters
🧹 Cleaned: {len(cleaned_text):,} characters
📝 Chunks: {len(chunks)} (avg {len(cleaned_text)//len(chunks):,} chars each)
💾 Saved: {output_path.name}
🔧 Method: {extraction_method}"""

        return raw_text, cleaned_text, status, gr.update(visible=True)

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return "", "", f"❌ Error processing PDF: {str(e)}", gr.update(visible=False)

def get_available_voices_enhanced():
    """Get enhanced voice list with metadata"""
    try:
        return voice_manager.get_voice_display_names()
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        return ["🔊 Standard Voice"]

def generate_audio_with_progress(text, voice_display_name, use_chunking, chunk_size):
    """Generate audio with progress tracking and enhanced error handling"""
    if not text.strip():
        return None, [], "❌ No text to convert to speech!", gr.update(visible=False)

    if not voice_display_name:
        return None, [], "❌ Please select a voice!", gr.update(visible=False)

    try:
        voice_id = voice_manager.get_voice_id_from_display(voice_display_name)

        if len(text) > config.MAX_TEXT_LENGTH:
            return None, [], f"❌ Text too long: {len(text)} characters (max {config.MAX_TEXT_LENGTH})", gr.update(visible=False)

        logger.info(f"Starting audio generation with voice: {voice_id}")

        if use_chunking:
            chunks = text_cleaner.split_into_chunks(text, chunk_size)
            logger.info(f"Generating audio for {len(chunks)} chunks")

            if len(chunks) > 50:
                return None, [], f"❌ Too many chunks: {len(chunks)} (reduce text or increase chunk size)", gr.update(visible=False)

            progress_info = {"completed": 0, "total": len(chunks)}

            def update_progress(ratio):
                progress_info["completed"] = int(ratio * progress_info["total"])

            audio_files = tts_generator.generate_audio_from_chunks(chunks, voice_id, update_progress)

            if audio_files:
                total_duration = 0
                try:
                    for audio_file in audio_files:
                        waveform, sample_rate = torchaudio.load(audio_file)
                        total_duration += waveform.shape[1] / sample_rate
                except:
                    total_duration = len(audio_files) * 30

                success_msg = f"""✅ Audio generation completed!
🎵 Generated: {len(audio_files)} files
📊 Total duration: ~{total_duration/60:.1f} minutes
🎭 Voice: {voice_display_name}
🔤 Chunks: {len(chunks)} (avg {len(text)//len(chunks):,} chars each)"""

                return audio_files[0], audio_files, success_msg, gr.update(visible=True)
            else:
                return None, [], "❌ Failed to generate any audio files!", gr.update(visible=False)
        else:
            if len(text) > 5000:
                warning_msg = f"⚠️ Text is quite long ({len(text)} chars) for single file. Consider chunking!"
                return None, [], warning_msg, gr.update(visible=False)

            audio_path = tts_generator.generate_audio(text, voice_id)

            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                duration = waveform.shape[1] / sample_rate
                duration_str = f"{duration/60:.1f} minutes"
            except:
                duration_str = "unknown"

            success_msg = f"""✅ Single audio file generated!
🎵 Duration: ~{duration_str}
🎭 Voice: {voice_display_name}
📄 Text length: {len(text):,} characters"""

            return audio_path, [audio_path], success_msg, gr.update(visible=True)

    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        error_msg = f"❌ Audio generation failed: {str(e)}"
        return None, [], error_msg, gr.update(visible=False)

def upload_voice_sample_enhanced(audio_file, voice_name):
    """Enhanced voice sample upload with validation"""
    if not audio_file or not voice_name.strip():
        return "❌ Please provide both audio file and voice name!", gr.update()

    try:
        voice_name = voice_name.strip()
        if len(voice_name) < 2:
            return "❌ Voice name too short (minimum 2 characters)!", gr.update()

        if len(voice_name) > 50:
            return "❌ Voice name too long (maximum 50 characters)!", gr.update()

        is_valid, validation_message = voice_manager.validate_voice_sample(audio_file)
        if not is_valid:
            return f"❌ {validation_message}", gr.update()

        voice_id = re.sub(r'[^\w\s\-]', '', voice_name.lower().replace(' ', '_'))
        voice_dir = Path(config.VOICE_SAMPLES_DIR) / voice_id
        voice_dir.mkdir(exist_ok=True)

        reference_path = voice_dir / "reference.wav"

        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            torchaudio.save(str(reference_path), waveform, sample_rate)
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return f"❌ Failed to process audio file: {e}", gr.update()

        duration = waveform.shape[1] / sample_rate
        if duration >= 20 and sample_rate >= 22050:
            quality = "high"
        elif duration >= 10 and sample_rate >= 16000:
            quality = "medium"
        else:
            quality = "low"

        success = voice_manager.add_voice(voice_id, voice_name,
                                        f"Custom voice ({duration:.1f}s)", quality)

        if success:
            new_choices = get_available_voices_enhanced()
            updated_dropdown = gr.Dropdown(choices=new_choices,
                                         value=f"🎭 {voice_name} ({quality})")

            success_msg = f"""✅ Voice '{voice_name}' uploaded successfully!
🎵 Duration: {duration:.1f} seconds
📊 Sample rate: {sample_rate} Hz
⭐ Quality: {quality}
💾 Saved as: {voice_id}"""

            return success_msg, updated_dropdown
        else:
            return "❌ Failed to save voice configuration!", gr.update()

    except Exception as e:
        logger.error(f"Voice upload failed: {e}")
        return f"❌ Error uploading voice: {str(e)}", gr.update()

def refresh_voices_enhanced():
    """Enhanced voice refresh with error handling"""
    try:
        choices = get_available_voices_enhanced()
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        logger.error(f"Error refreshing voices: {e}")
        return gr.Dropdown(choices=["🔊 Standard Voice"], value="🔊 Standard Voice")

def cleanup_files():
    """Clean up old generated files"""
    try:
        tts_generator.cleanup_old_files()
        return "✅ Cleanup completed successfully!"
    except Exception as e:
        return f"❌ Cleanup failed: {e}"

# --- Text Editing Functions ---
def update_text_stats(text):
    """Update text statistics in real-time"""
    if not text:
        return 0, 0, "0 minutes"

    words = TextEditor.count_words(text)
    chars = TextEditor.count_characters(text)
    reading_time = TextEditor.estimate_reading_time(text)
    return words, chars, reading_time

def clear_text():
    """Clear the text editor"""
    return ""

def find_and_highlight(text, find_text):
    """Find text and return info about matches"""
    if not text or not find_text:
        return text, "No search text provided."

    matches = TextEditor.find_text_matches(text, find_text)
    if matches > 0:
        return text, f"✅ Found {matches} matches for '{find_text}'"
    else:
        return text, f"❌ No matches found for '{find_text}'"

def replace_all_text(text, find_text, replace_text):
    """Replace all occurrences of find_text with replace_text"""
    if not text or not find_text:
        return text, "❌ Please provide both text to find and replacement text."

    new_text, count = TextEditor.replace_text(text, find_text, replace_text)
    if count > 0:
        return new_text, f"✅ Replaced {count} occurrences of '{find_text}' with '{replace_text}'"
    else:
        return text, f"❌ No matches found for '{find_text}'"

def sync_text_to_preview(text):
    """Sync text from editor to preview"""
    return text

def load_text_file(file_path):
    """Load text from uploaded file"""
    if not file_path:
        return "", "❌ No file selected."

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, f"✅ Loaded {len(content):,} characters from {os.path.basename(file_path)}"
    except Exception as e:
        return "", f"❌ Error loading file: {str(e)}"

def save_text_to_file(text):
    """Save current text to a file"""
    if not text:
        return "❌ No text to save!"

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"edited_text_{timestamp}.txt"
        filepath = os.path.join(config.PROCESSED_PDFS_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        return f"✅ Text saved to {filename} ({len(text):,} characters)"
    except Exception as e:
        return f"❌ Error saving file: {str(e)}"

def clean_text_formatting(text):
    """Clean common text formatting issues"""
    if not text:
        return "", "❌ No text to clean."

    original_length = len(text)
    cleaned_text = TextEditor.clean_formatting(text)

    return cleaned_text, f"✅ Text formatting cleaned! {original_length:,} → {len(cleaned_text):,} characters"

# --- Enhanced Gradio Interface ---
def create_gradio_interface():
    """Create the enhanced Gradio interface with Ollama integration"""
    with gr.Blocks(
        title="Enhanced PDF to TTS Converter with AI Processing",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .status-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
        .status-error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
        .status-warning { background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }
        .text-editor { font-family: 'Consolas', 'Monaco', 'Courier New', monospace; }
        .edit-highlight { background-color: #fff3cd; border: 2px solid #ffc107; }
        .stats-panel { background-color: #f8f9fa; padding: 10px; border-radius: 8px; }
        .ollama-section { background-color: #e8f5e8; padding: 15px; border-radius: 10px; border: 2px solid #28a745; }
        """
    ) as demo:

        gr.Markdown("# 🤖 Enhanced PDF to TTS Converter with AI Processing")
        gr.Markdown("**Extract text from PDFs, process with AI, edit with powerful tools, and convert to high-quality speech**")

        with gr.Tab("📄 PDF Processing & AI Enhancement"):
            gr.Markdown("## Upload PDF and Process with AI")

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_input = gr.File(
                        label="📁 Upload PDF File",
                        file_types=[".pdf"],
                        type="filepath",
                        height=150
                    )

                with gr.Column(scale=1):
                    extraction_method = gr.Dropdown(
                        choices=["auto", "pdfplumber", "pymupdf", "pypdf2"],
                        value="auto",
                        label="🔧 Extraction Method",
                        info="Auto tries methods in order of quality"
                    )

                    preserve_structure = gr.Checkbox(
                        label="📐 Preserve Structure",
                        value=True,
                        info="Keep paragraph breaks"
                    )

                    chunk_size_preview = gr.Slider(
                        minimum=config.MIN_CHUNK_SIZE,
                        maximum=config.MAX_CHUNK_SIZE,
                        value=1000,
                        step=100,
                        label="📏 Preview Chunk Size",
                        info="For estimation only"
                    )

            process_btn = gr.Button("📝 Extract Text from PDF", variant="primary", size="lg")

            processing_status = gr.Textbox(
                label="📊 Processing Status",
                lines=6,
                placeholder="Upload a PDF and click 'Extract Text from PDF' to begin...",
                show_label=True
            )

            with gr.Row():
                with gr.Column(scale=1):
                    raw_text = gr.Textbox(
                        label="📄 Raw Extracted Text",
                        lines=15,
                        placeholder="Raw text from PDF will appear here...",
                        show_label=True
                    )
                with gr.Column(scale=1):
                    extracted_text = gr.Textbox(
                        label="🧹 Cleaned Text",
                        lines=15,
                        placeholder="Cleaned text will appear here and can be edited...",
                        show_label=True
                    )

            # AI Processing Section
            with gr.Group(elem_classes=["ollama-section"]):
                gr.Markdown("### 🤖 **AI-Powered Document Processing with Ollama**")
                gr.Markdown("*Process your extracted text with AI for better understanding and TTS optimization*")

                # Ollama Connection Status
                ollama_status = gr.Textbox(
                    label="🔗 Ollama Connection Status",
                    value="Click 'Check Connection' to verify Ollama availability",
                    interactive=False,
                    lines=2
                )

                with gr.Row():
                    check_connection_btn = gr.Button("🔗 Check Ollama Connection", size="sm")
                    available_models = gr.Dropdown(
                        choices=get_available_ollama_models(),
                        value=config.DEFAULT_MODEL,
                        label="🤖 Ollama Model",
                        info="Select AI model for processing"
                    )
                    refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

                with gr.Row():
                    with gr.Column(scale=2):
                        ai_task_type = gr.Dropdown(
                            choices=["summarize", "analyze", "extract", "simplify", "outline", "custom"],
                            value="summarize",
                            label="🎯 AI Processing Task",
                            info="Choose what you want the AI to do"
                        )

                    with gr.Column(scale=2):
                        custom_query = gr.Textbox(
                            label="❓ Custom Query",
                            placeholder="Enter your custom question (only for 'custom' task)",
                            max_lines=2,
                            visible=False
                        )

                    with gr.Column(scale=1):
                        use_ai_cache = gr.Checkbox(
                            label="💾 Use Cache",
                            value=True,
                            info="Cache AI results for faster processing"
                        )

                process_ai_btn = gr.Button("🤖 Process with AI", variant="primary", size="lg")

                ai_processed_text = gr.Textbox(
                    label="🤖 AI Processed Text",
                    lines=20,
                    placeholder="AI-processed text will appear here and can be edited before TTS conversion...",
                    show_label=True,
                    elem_classes=["text-editor"]
                )

                ai_status = gr.Textbox(
                    label="📊 AI Processing Status",
                    lines=8,
                    placeholder="AI processing results and statistics will appear here...",
                    interactive=False
                )

            # Enhanced text editing section for AI-processed text
            with gr.Group():
                gr.Markdown("### ✏️ **Text Editor for AI-Processed Content**")
                gr.Markdown("*Fine-tune your AI-processed text before converting to speech*")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📊 **Text Statistics**")

                        ai_word_count = gr.Number(
                            label="📝 Words",
                            value=0,
                            interactive=False
                        )

                        ai_char_count = gr.Number(
                            label="🔤 Characters",
                            value=0,
                            interactive=False
                        )

                        ai_reading_time = gr.Textbox(
                            label="⏰ Reading Time",
                            value="0 minutes",
                            interactive=False
                        )

                        gr.Markdown("#### 🛠️ **Quick Actions**")
                        clean_ai_format_btn = gr.Button("🧹 Clean Format", size="sm", variant="secondary")
                        clear_ai_btn = gr.Button("🗑️ Clear All", size="sm", variant="stop")
                        save_ai_btn = gr.Button("💾 Save", size="sm")

                    with gr.Column(scale=3):
                        gr.Markdown("#### 🔍 **Find & Replace Tools**")

                        with gr.Row():
                            with gr.Column(scale=2):
                                ai_find_text = gr.Textbox(
                                    label="🔍 Find Text",
                                    placeholder="Enter text to find...",
                                    max_lines=1
                                )

                            with gr.Column(scale=2):
                                ai_replace_text = gr.Textbox(
                                    label="🔄 Replace With",
                                    placeholder="Enter replacement text...",
                                    max_lines=1
                                )

                            with gr.Column(scale=1):
                                ai_find_btn = gr.Button("🔍 Find", size="sm")
                                ai_replace_btn = gr.Button("🔄 Replace All", size="sm", variant="secondary")

            text_stats = gr.HTML(visible=False)

        with gr.Tab("🎙️ Audio Generation"):
            gr.Markdown("## Convert Your Processed Text to High-Quality Speech")

            # Text source selection
            with gr.Group():
                gr.Markdown("### 📝 **Select Text Source**")
                text_source = gr.Radio(
                    choices=["🧹 Use Cleaned Text", "🤖 Use AI-Processed Text"],
                    value="🤖 Use AI-Processed Text",
                    label="Text Source",
                    info="Choose which processed text to convert to audio"
                )

                text_preview = gr.Textbox(
                    label="Text to Convert",
                    lines=12,
                    interactive=False,
                    placeholder="Selected text will appear here..."
                )

                sync_text_btn = gr.Button("🔄 Load Selected Text", size="sm", variant="secondary")

            with gr.Row():
                with gr.Column(scale=2):
                    voice_dropdown = gr.Dropdown(
                        label="🎭 Select Voice",
                        choices=get_available_voices_enhanced(),
                        value=get_available_voices_enhanced()[0] if get_available_voices_enhanced() else None,
                        info="Choose from standard or custom cloned voices"
                    )

                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")

            with gr.Row():
                with gr.Column():
                    use_chunking = gr.Checkbox(
                        label="✂️ Split into Chunks",
                        value=True,
                        info="Recommended for long texts (>2000 chars)"
                    )

                    chunk_size = gr.Slider(
                        minimum=config.MIN_CHUNK_SIZE,
                        maximum=config.MAX_CHUNK_SIZE,
                        value=1000,
                        step=100,
                        label="📏 Chunk Size (characters)",
                        info="Smaller chunks = more files, better memory usage"
                    )

                with gr.Column():
                    estimated_chunks = gr.Number(
                        label="📊 Estimated Chunks",
                        value=0,
                        interactive=False
                    )

                    estimated_duration = gr.Textbox(
                        label="⏱️ Estimated Duration",
                        value="0 minutes",
                        interactive=False
                    )

            audio_btn = gr.Button("🔊 Generate Audio from Processed Text", variant="primary", size="lg")

            with gr.Row():
                first_audio = gr.Audio(
                    label="🎵 Preview (First Audio File)",
                    type="filepath"
                )

            all_audio_files = gr.File(
                label="📁 All Generated Audio Files",
                file_count="multiple",
                type="filepath"
            )

            audio_status = gr.Textbox(
                label="📊 Generation Status",
                lines=6,
                placeholder="Process your text, then click 'Generate Audio'..."
            )

            download_section = gr.HTML(visible=False)

        with gr.Tab("🎭 Voice Management"):
            gr.Markdown("## Upload and Manage Custom Voice Samples")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 Upload New Voice")
                    upload_audio = gr.File(
                        label="🎵 Audio Sample",
                        file_types=config.SUPPORTED_AUDIO_FORMATS,
                        type="filepath"
                    )

                    upload_name = gr.Textbox(
                        label="🏷️ Voice Name",
                        placeholder="Enter a unique name for this voice",
                        max_lines=1
                    )

                    upload_btn = gr.Button("⬆️ Upload Voice", variant="primary")
                    upload_status = gr.Textbox(label="📊 Upload Status", lines=4)

                with gr.Column():
                    gr.Markdown("### 📋 Voice Requirements")
                    gr.Markdown("""
                    **For optimal results:**
                    - 📏 **Duration**: 10-30 seconds
                    - 🎤 **Quality**: Clear, noise-free recording
                    - 🗣️ **Speaker**: Single person speaking naturally
                    - 🎵 **Format**: WAV, MP3, FLAC, or OGG
                    - 💾 **Size**: 100KB - 50MB
                    - 🎯 **Content**: Natural conversation works best
                    - 📊 **Sample Rate**: 16kHz+ recommended
                    """)

            with gr.Row():
                cleanup_btn = gr.Button("🧹 Cleanup Old Files", variant="secondary")
                cleanup_status = gr.Textbox(label="Cleanup Status", lines=1)

        with gr.Tab("🤖 Ollama Settings"):
            gr.Markdown("## Configure Ollama AI Processing")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔧 **Ollama Configuration**")

                    current_ollama_url = gr.Textbox(
                        label="🌐 Ollama Base URL",
                        value=config.OLLAMA_BASE_URL,
                        info="URL where Ollama is running"
                    )

                    current_model = gr.Textbox(
                        label="🤖 Current Model",
                        value=config.DEFAULT_MODEL,
                        interactive=False,
                        info="Currently selected model"
                    )

                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config.OLLAMA_TEMPERATURE,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Controls randomness (0.0 = deterministic, 1.0 = creative)"
                    )

                    max_tokens_slider = gr.Slider(
                        minimum=1000,
                        maximum=8000,
                        value=config.OLLAMA_MAX_TOKENS,
                        step=500,
                        label="📊 Max Tokens",
                        info="Maximum context window size"
                    )

                with gr.Column():
                    gr.Markdown("### 📖 **Task Descriptions**")
                    gr.Markdown("""
                    **Available AI Processing Tasks:**

                    - **📋 Summarize**: Create a concise summary of the main points
                    - **🔍 Analyze**: Deep analysis of themes, arguments, and conclusions
                    - **📤 Extract**: Pull out key facts, figures, and important quotes
                    - **✨ Simplify**: Rewrite in simpler language for general audiences
                    - **📝 Outline**: Create a structured outline of the document
                    - **❓ Custom**: Answer specific questions about the content

                    **Best Practices:**
                    - Use **Summarize** for long documents you want to condense
                    - Use **Analyze** for research papers and complex content
                    - Use **Extract** to pull key information for quick reference
                    - Use **Simplify** to make technical content more accessible
                    - Use **Custom** with specific questions about the content
                    """)

            update_config_btn = gr.Button("💾 Update Ollama Settings", variant="secondary")
            config_status = gr.Textbox(label="Configuration Status", lines=2)

        with gr.Tab("ℹ️ Help & Guide"):
            gr.Markdown("## 📖 Complete Usage Guide")

            with gr.Accordion("🚀 Quick Start Guide", open=True):
                gr.Markdown("""
                ### **Complete Workflow:**
                1. **📄 Upload PDF**: Go to 'PDF Processing' tab and upload your document
                2. **🔧 Extract**: Click 'Extract Text from PDF' and review the cleaned text
                3. **🤖 AI Processing**: Choose an AI task (summarize, analyze, etc.) and click 'Process with AI'
                4. **✏️ Edit**: Fine-tune the AI-processed text using the editing tools
                5. **🎭 Select Voice**: Go to 'Audio Generation' tab and choose a voice
                6. **⚙️ Configure**: Set up chunking options and text source
                7. **🔊 Generate**: Click 'Generate Audio' and wait for completion
                8. **💾 Download**: Save the generated audio files
                """)

            with gr.Accordion("🤖 AI Processing Features", open=True):
                gr.Markdown("""
                **Ollama AI Integration:**
                - **🧠 Smart Processing**: Uses advanced AI models to understand and transform your content
                - **📋 Multiple Tasks**: Summarize, analyze, extract, simplify, outline, or custom queries
                - **💾 Intelligent Caching**: Speeds up repeated processing with smart caching
                - **🔄 Model Selection**: Choose from available Ollama models for different capabilities
                - **⚡ Progress Tracking**: Real-time feedback during AI processing
                - **💡 Optimization**: Content is optimized for better text-to-speech conversion

                **Task Examples:**
                - **Summarize**: "Provide a comprehensive summary suitable for audio presentation"
                - **Analyze**: "What are the main arguments and supporting evidence?"
                - **Custom**: "What are the key recommendations for implementation?"
                """)

            with gr.Accordion("✏️ Advanced Text Editing", open=True):
                gr.Markdown("""
                **Powerful editing tools for both cleaned and AI-processed text:**
                - **📊 Live Statistics**: Real-time word count, character count, and reading time
                - **🔍 Smart Search**: Find and replace with case-insensitive matching
                - **🧹 Auto-Formatting**: Clean spacing, punctuation, and structure
                - **💾 File Operations**: Save edits and load external text files
                - **🔄 Text Synchronization**: Seamlessly move between processing and audio tabs
                - **📝 Direct Editing**: Click in any text area to make manual changes
                """)

            with gr.Accordion("🔧 Setup Requirements", open=False):
                gr.Markdown(f"""
                **Required Dependencies:**
                ```bash
                pip install PyPDF2 PyMuPDF pdfplumber speechbrain TTS num2words
                pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
                pip install gradio requests
                ```

                **Ollama Setup:**
                1. Install Ollama: Visit https://ollama.ai
                2. Start Ollama service: `ollama serve`
                3. Pull a model: `ollama pull llama2`
                4. Verify connection in the app

                **Current Configuration:**
                - 📁 Max PDF size: 100MB
                - 🎵 Max audio size: 50MB
                - 📝 Max text length: {config.MAX_TEXT_LENGTH:,} characters
                - 🔄 Concurrent tasks: {config.MAX_CONCURRENT_TASKS}
                - 🤖 Ollama URL: {config.OLLAMA_BASE_URL}
                """)

        # Event Handlers

        # PDF Processing
        process_btn.click(
            fn=process_pdf_enhanced,
            inputs=[pdf_input, extraction_method, preserve_structure, chunk_size_preview],
            outputs=[raw_text, extracted_text, processing_status, text_stats]
        )

        # Ollama Connection and Model Management
        def update_connection_status():
            return check_ollama_status()

        def refresh_ollama_models():
            models = get_available_ollama_models()
            return gr.Dropdown(choices=models, value=models[0] if models else config.DEFAULT_MODEL)

        def update_model_selection(model_name):
            status = update_ollama_model(model_name)
            return status

        check_connection_btn.click(
            fn=update_connection_status,
            outputs=[ollama_status]
        )

        refresh_models_btn.click(
            fn=refresh_ollama_models,
            outputs=[available_models]
        )

        available_models.change(
            fn=update_model_selection,
            inputs=[available_models],
            outputs=[ollama_status]
        )

        # Show/hide custom query based on task type
        def update_custom_query_visibility(task_type):
            return gr.update(visible=task_type == "custom")

        ai_task_type.change(
            fn=update_custom_query_visibility,
            inputs=[ai_task_type],
            outputs=[custom_query]
        )

        # AI Processing
        process_ai_btn.click(
            fn=process_with_ollama,
            inputs=[extracted_text, ai_task_type, custom_query, use_ai_cache],
            outputs=[ai_processed_text, ai_status]
        )

        # Text editing for AI processed text
        ai_processed_text.change(
            fn=update_text_stats,
            inputs=[ai_processed_text],
            outputs=[ai_word_count, ai_char_count, ai_reading_time]
        )

        # AI text editing buttons
        clear_ai_btn.click(
            fn=clear_text,
            outputs=ai_processed_text
        )

        clean_ai_format_btn.click(
            fn=clean_text_formatting,
            inputs=[ai_processed_text],
            outputs=[ai_processed_text, ai_status]
        )

        ai_find_btn.click(
            fn=find_and_highlight,
            inputs=[ai_processed_text, ai_find_text],
            outputs=[ai_processed_text, ai_status]
        )

        ai_replace_btn.click(
            fn=replace_all_text,
            inputs=[ai_processed_text, ai_find_text, ai_replace_text],
            outputs=[ai_processed_text, ai_status]
        )

        save_ai_btn.click(
            fn=save_text_to_file,
            inputs=[ai_processed_text],
            outputs=[ai_status]
        )

        # Text source selection for audio generation
        def update_text_preview(source, cleaned_text, ai_text):
            if source == "🧹 Use Cleaned Text":
                return cleaned_text
            else:
                return ai_text

        sync_text_btn.click(
            fn=update_text_preview,
            inputs=[text_source, extracted_text, ai_processed_text],
            outputs=[text_preview]
        )

        # Auto-update preview when source changes
        text_source.change(
            fn=update_text_preview,
            inputs=[text_source, extracted_text, ai_processed_text],
            outputs=[text_preview]
        )

        # Auto-sync when texts change
        extracted_text.change(
            fn=lambda source, cleaned, ai: update_text_preview(source, cleaned, ai),
            inputs=[text_source, extracted_text, ai_processed_text],
            outputs=[text_preview]
        )

        ai_processed_text.change(
            fn=lambda source, cleaned, ai: update_text_preview(source, cleaned, ai),
            inputs=[text_source, extracted_text, ai_processed_text],
            outputs=[text_preview]
        )

        # Audio generation uses the preview text
        audio_btn.click(
            fn=generate_audio_with_progress,
            inputs=[text_preview, voice_dropdown, use_chunking, chunk_size],
            outputs=[first_audio, all_audio_files, audio_status, download_section]
        )

        # Voice management
        refresh_btn.click(
            fn=refresh_voices_enhanced,
            outputs=voice_dropdown
        )

        upload_btn.click(
            fn=upload_voice_sample_enhanced,
            inputs=[upload_audio, upload_name],
            outputs=[upload_status, voice_dropdown]
        )

        cleanup_btn.click(
            fn=cleanup_files,
            outputs=cleanup_status
        )

        # Update estimates when text or settings change
        def update_estimates(text, chunk_size, use_chunking):
            """Update chunk and duration estimates"""
            if not text or not use_chunking:
                return 0, "0 minutes"

            chunks = text_cleaner.split_into_chunks(text, chunk_size)
            estimated_duration_min = len(chunks) * 2  # Rough estimate: 2 min per chunk
            duration_str = f"~{estimated_duration_min} minutes"

            return len(chunks), duration_str

        for component in [text_preview, chunk_size, use_chunking]:
            component.change(
                fn=update_estimates,
                inputs=[text_preview, chunk_size, use_chunking],
                outputs=[estimated_chunks, estimated_duration]
            )

        # Ollama Settings Tab
        def update_ollama_config(url, temperature, max_tokens):
            """Update Ollama configuration"""
            try:
                global ollama_processor
                ollama_processor.base_url = url.rstrip('/')
                ollama_processor.api_url = f"{ollama_processor.base_url}/api/generate"
                ollama_processor.temperature = temperature
                ollama_processor.max_tokens = max_tokens

                # Update global config
                config.OLLAMA_BASE_URL = url
                config.OLLAMA_TEMPERATURE = temperature
                config.OLLAMA_MAX_TOKENS = max_tokens

                # Test connection with new settings
                is_connected, message = ollama_processor.check_ollama_connection()

                if is_connected:
                    return f"✅ Configuration updated successfully!\n{message}"
                else:
                    return f"⚠️ Configuration updated but connection failed:\n{message}"

            except Exception as e:
                return f"❌ Error updating configuration: {str(e)}"

        update_config_btn.click(
            fn=update_ollama_config,
            inputs=[current_ollama_url, temperature_slider, max_tokens_slider],
            outputs=[config_status]
        )

    return demo

# --- Main Application ---
if __name__ == "__main__":
    # Setup logging
    logger.info("🚀 Starting Enhanced PDF to TTS Converter with AI Processing...")
    logger.info(f"📁 Voice samples directory: {config.VOICE_SAMPLES_DIR}")
    logger.info(f"🔊 Generated audio directory: {config.GENERATED_AUDIO_DIR}")
    logger.info(f"📄 Processed PDFs directory: {config.PROCESSED_PDFS_DIR}")
    logger.info(f"💾 Cache directory: {config.CACHE_DIR}")
    logger.info(f"🤖 Ollama cache directory: {config.OLLAMA_CACHE_DIR}")
    logger.info(f"🌐 Ollama base URL: {config.OLLAMA_BASE_URL}")

    # Check for required dependencies
    required_packages = {
        'PyPDF2': 'PyPDF2',
        'fitz': 'PyMuPDF',
        'pdfplumber': 'pdfplumber',
        'speechbrain': 'speechbrain',
        'TTS': 'TTS',
        'num2words': 'num2words',
        'torchaudio': 'torchaudio',
        'requests': 'requests'
    }

    missing_packages = []

    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        logger.warning("⚠️  Missing required packages. Install with:")
        logger.warning(f"pip install {' '.join(missing_packages)}")
        logger.warning("\nSpecific installation commands:")
        logger.warning("pip install PyPDF2 PyMuPDF pdfplumber speechbrain TTS num2words requests")
        logger.warning("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("\n" + "="*60)
        print("⚠️  MISSING DEPENDENCIES DETECTED!")
        print("="*60)
        print("The following packages need to be installed:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall command:")
        print(f"pip install {' '.join(missing_packages)}")
        print("="*60)

    # Check Ollama connection at startup
    logger.info("🤖 Checking Ollama connection...")
    try:
        is_connected, connection_message = ollama_processor.check_ollama_connection()
        if is_connected:
            logger.info(f"✅ {connection_message}")
        else:
            logger.warning(f"⚠️ {connection_message}")
            logger.warning("📝 Note: Ollama features will be disabled until connection is established")
    except Exception as e:
        logger.error(f"❌ Error checking Ollama: {e}")

    try:
        # Create the Gradio interface
        demo = create_gradio_interface()

        # Launch the application
        logger.info("🌐 Launching enhanced web interface with AI processing...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=1602,
            share=False,
            show_error=True,
            quiet=False,
            max_threads=config.MAX_CONCURRENT_TASKS * 2
        )

    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed to start: {e}")
        raise
    finally:
        # Cleanup on exit
        try:
            tts_generator.cleanup_old_files(max_age_hours=1)
            logger.info("🧹 Final cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")