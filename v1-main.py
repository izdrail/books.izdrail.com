

import os

import re
import uuid
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import hashlib
import logging
import traceback
from dataclasses import dataclass
import requests

# Third-party imports
import gradio as gr
import PyPDF2
import fitz  # PyMuPDF
import torchaudio
from TTS.api import TTS
from num2words import num2words
def diagnose_pdf_file(pdf_file):
    """Diagnose PDF file for debugging"""
    if not pdf_file:
        return "❌ No PDF file provided"

    try:
        diagnosis = pdf_extractor.diagnose_pdf(pdf_file.name)

        report = f"""📄 PDF Diagnosis Report for: {os.path.basename(pdf_file.name)}

📁 **File Info:**
- Exists: {'✅' if diagnosis['file_exists'] else '❌'}
- Size: {diagnosis['file_size'] / 1024 / 1024:.2f} MB
- Readable: {'✅' if diagnosis['is_readable'] else '❌'}

🔧 **PyMuPDF Status:**
- Can Open: {'✅' if diagnosis['pymupdf_info'].get('can_open', False) else '❌'}
- Page Count: {diagnosis['pymupdf_info'].get('page_count', 'N/A')}
- Encrypted: {'⚠️' if diagnosis['pymupdf_info'].get('is_encrypted', False) else '✅'}
- Can Extract Text: {'✅' if diagnosis['pymupdf_info'].get('can_extract_text', False) else '❌'}

🔧 **PyPDF2 Status:**
- Can Open: {'✅' if diagnosis['pypdf2_info'].get('can_open', False) else '❌'}
- Page Count: {diagnosis['pypdf2_info'].get('page_count', 'N/A')}
- Encrypted: {'⚠️' if diagnosis['pypdf2_info'].get('is_encrypted', False) else '✅'}
- Can Extract Text: {'✅' if diagnosis['pypdf2_info'].get('can_extract_text', False) else '❌'}

💡 **Recommendation:** {diagnosis['recommended_method']}

⚠️ **Issues Found:**
{chr(10).join(f"- {issue}" for issue in diagnosis['issues']) if diagnosis['issues'] else 'None'}
"""
        return report

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        return f"❌ Diagnosis failed: {str(e)}"


def count_text_stats(text):
    """Count text statistics"""
    try:
        if not text:
            return 0, 0, "0 minutes"

        words = len(text.split())
        chars = len(text)
        # Estimate 150 words per minute reading speed
        minutes = max(1, words // 150)

        if minutes == 1:
            time_str = "1 minute"
        elif minutes < 60:
            time_str = f"{minutes} minutes"
        else:
            hours = minutes // 60
            remaining_mins = minutes % 60
            time_str = f"{hours}h {remaining_mins}m"

        return words, chars, time_str
    except Exception as e:
        logger.error(f"Error counting text stats: {e}")
        return 0, 0, "Error"

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more info
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('pdf_tts_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Configuration ---
@dataclass
class Config:
    """Simplified application configuration"""
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    DEFAULT_MODEL: str = "llama2"
    VOICE_SAMPLES_DIR: str = "voice_samples"
    GENERATED_AUDIO_DIR: str = "generated_audio"
    PROCESSED_PAGES_DIR: str = "processed_pages"
    CACHE_DIR: str = "cache"
    STANDARD_VOICE_NAME: str = "standard"
    MAX_PAGES_BATCH: int = 5
    SUPPORTED_AUDIO_FORMATS: List[str] = None
    OLLAMA_TEMPERATURE: float = 0.3
    TTS_SUMMARY_PROMPT: str = """Please rewrite this page content to be clear, engaging, and perfect for text-to-speech conversion. 
Make it flow naturally when spoken aloud, remove any formatting artifacts, and maintain all important information.
Keep it conversational and easy to listen to:

{page_content}

TTS-Optimized Version:"""

    def __post_init__(self):
        if self.SUPPORTED_AUDIO_FORMATS is None:
            self.SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.ogg']

        # Create directories
        for directory in [self.VOICE_SAMPLES_DIR, self.GENERATED_AUDIO_DIR,
                          self.PROCESSED_PAGES_DIR, self.CACHE_DIR]:
            Path(directory).mkdir(exist_ok=True)


config = Config()


# --- Ollama Integration ---
class OllamaProcessor:
    """Simplified Ollama processor for TTS summaries"""

    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL.rstrip('/')
        self.model_name = config.DEFAULT_MODEL
        self.api_url = f"{self.base_url}/api/generate"
        self.cache_dir = Path(config.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

    def check_connection(self) -> tuple[bool, str]:
        """Check if Ollama is available"""
        try:
            logger.debug(f"Checking Ollama connection at {self.base_url}")
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                logger.debug(f"Available models: {models}")
                if not models:
                    return False, "⚠️ Ollama running but no models available. Run: ollama pull llama2"

                model_names = [m.get('name', '') for m in models if isinstance(m, dict)]
                logger.debug(f"Model names: {model_names}")
                if not any(self.model_name in name for name in model_names):
                    return False, f"⚠️ Model '{self.model_name}' not found. Available: {', '.join(model_names)}"

                return True, f"✅ Connected to Ollama with model: {self.model_name}"
            else:
                return False, f"❌ Ollama connection failed: HTTP {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"❌ Cannot connect to Ollama at {self.base_url}. Is it running?"
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            return False, f"❌ Connection error: {str(e)}"

    def get_cache_key(self, text: str) -> str:
        """Generate cache key for processed text"""
        try:
            return hashlib.md5(f"{text}_{self.model_name}".encode()).hexdigest()
        except Exception as e:
            logger.error(f"Cache key generation failed: {e}")
            return str(hash(text))[:16]  # Fallback

    def load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load processed text from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.txt"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.debug(f"Using cached result: {cache_key[:8]}...")
                    return content
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def save_to_cache(self, cache_key: str, text: str):
        """Save processed text to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.txt"
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.debug(f"Cached result: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def process_page_for_tts(self, page_content: str, use_cache: bool = True) -> tuple[bool, str]:
        """Process a single page for TTS optimization"""
        try:
            if not page_content or not page_content.strip():
                return False, "Empty page content"

            logger.debug(f"Processing page content: {len(page_content)} chars")

            # Check cache first
            if use_cache:
                cache_key = self.get_cache_key(page_content)
                cached_result = self.load_from_cache(cache_key)
                if cached_result:
                    return True, cached_result

            # Prepare prompt - be extra careful with string formatting
            try:
                prompt = config.TTS_SUMMARY_PROMPT.format(page_content=page_content)
            except Exception as e:
                logger.error(f"Prompt formatting failed: {e}")
                # Fallback without formatting
                prompt = f"Please rewrite this text for TTS: {page_content}"

            # Send to Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.OLLAMA_TEMPERATURE,
                    "num_predict": 1024,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }

            logger.debug(f"Sending request to Ollama: {len(prompt)} chars")
            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                result = response.json()
                processed_text = result.get("response", "").strip()

                if processed_text:
                    # Cache successful result
                    if use_cache:
                        self.save_to_cache(cache_key, processed_text)
                    return True, processed_text
                else:
                    return False, "Empty response from Ollama"
            else:
                logger.error(f"Ollama HTTP error: {response.status_code} - {response.text}")
                return False, f"Ollama error: HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return False, "Ollama request timed out"
        except Exception as e:
            logger.error(f"Ollama processing error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Processing error: {str(e)}"


# --- PDF Page Extractor with Enhanced Debug ---
class PDFPageExtractor:
    """Extract PDF content page by page with comprehensive error handling"""

    def __init__(self):
        self.extraction_methods = ['pymupdf', 'pypdf2']

    def validate_pdf(self, pdf_path: str) -> tuple[bool, str]:
        """Validate PDF file"""
        try:
            logger.debug(f"Validating PDF: {pdf_path}")

            if not pdf_path:
                return False, "No PDF file provided"

            if not os.path.exists(pdf_path):
                return False, f"File not found: {pdf_path}"

            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
            logger.debug(f"PDF file size: {file_size:.2f}MB")

            if file_size > 100:
                return False, f"File too large: {file_size:.1f}MB (max 100MB)"

            # Try to open with both methods
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    page_count = len(reader.pages)
                    logger.debug(f"PyPDF2 page count: {page_count}")
                    if page_count == 0:
                        return False, "PDF contains no pages"
                return True, f"Valid PDF with {page_count} pages"
            except Exception as e:
                logger.warning(f"PyPDF2 validation failed: {e}, trying PyMuPDF...")
                try:
                    doc = fitz.open(pdf_path)
                    page_count = doc.page_count
                    logger.debug(f"PyMuPDF page count: {page_count}")
                    doc.close()
                    if page_count == 0:
                        return False, "PDF contains no pages"
                    return True, f"Valid PDF with {page_count} pages"
                except Exception as e2:
                    logger.error(f"Both PDF validation methods failed: PyPDF2: {e}, PyMuPDF: {e2}")
                    return False, f"Invalid PDF: {e2}"
        except Exception as e:
            logger.error(f"PDF validation error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Validation error: {e}"

    def get_page_count(self, pdf_path: str) -> int:
        """Get total number of pages with enhanced error handling"""
        try:
            logger.debug(f"Getting page count for: {pdf_path}")

            # Try PyMuPDF first (usually more reliable)
            try:
                doc = fitz.open(pdf_path)
                count = doc.page_count
                doc.close()
                logger.debug(f"PyMuPDF page count: {count}")
                if count > 0:
                    return count
                else:
                    logger.warning("PyMuPDF returned 0 pages, trying PyPDF2...")
            except Exception as e:
                logger.warning(f"PyMuPDF page count failed: {e}")

            # Fallback to PyPDF2 with multiple methods
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)

                    # Try multiple methods to get page count
                    count = 0
                    try:
                        count = len(reader.pages)
                        logger.debug(f"PyPDF2 page count (len): {count}")
                    except Exception as e:
                        logger.warning(f"reader.pages length failed: {e}")
                        try:
                            count = reader.numPages if hasattr(reader, 'numPages') else 0
                            logger.debug(f"PyPDF2 page count (numPages): {count}")
                        except Exception as e2:
                            logger.warning(f"reader.numPages failed: {e2}")

                            # Last resort: try to count manually
                            try:
                                count = 0
                                for _ in reader.pages:
                                    count += 1
                                logger.debug(f"PyPDF2 page count (manual): {count}")
                            except Exception as e3:
                                logger.error(f"Manual page counting failed: {e3}")

                    return max(0, count)

            except Exception as e:
                logger.warning(f"PyPDF2 page count failed: {e}")

        except Exception as e:
            logger.error(f"Failed to get page count: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        logger.warning("All page count methods failed, returning 0")
        return 0

    def extract_page_pymupdf(self, pdf_path: str, page_num: int) -> Tuple[bool, str]:
        """Extract single page using PyMuPDF with comprehensive error handling"""
        doc = None
        try:
            logger.debug(f"PyMuPDF extracting page {page_num + 1} from {pdf_path}")

            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            logger.debug(f"PDF has {total_pages} pages, requesting page {page_num + 1}")

            # Validate page number with extra safety checks
            if page_num < 0:
                return False, f"Invalid page number: {page_num + 1} (negative)"
            if page_num >= total_pages:
                return False, f"Page {page_num + 1} does not exist (PDF has {total_pages} pages)"

            # Additional safety check before accessing page
            try:
                if page_num >= len(doc):
                    return False, f"Page index {page_num} out of range (doc length: {len(doc)})"

                page = doc.load_page(page_num)  # Use load_page instead of direct indexing
                text = page.get_text(sort=True)  # Add sort=True for better text extraction
                logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")

                return True, text.strip()

            except (IndexError, ValueError, RuntimeError) as e:
                logger.error(f"PyMuPDF page access error for page {page_num + 1}: {e}")
                # Try alternative access method
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    return True, text.strip() if text else ""
                except Exception as e2:
                    logger.error(f"Alternative PyMuPDF access also failed: {e2}")
                    return False, f"Cannot access page {page_num + 1}: {e}"

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for page {page_num + 1}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"PyMuPDF extraction error: {e}"
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass

    def extract_page_pypdf2(self, pdf_path: str, page_num: int) -> Tuple[bool, str]:
        """Extract single page using PyPDF2 with comprehensive error handling"""
        try:
            logger.debug(f"PyPDF2 extracting page {page_num + 1} from {pdf_path}")

            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)

                # Multiple ways to check page count for robustness
                try:
                    total_pages = len(reader.pages)
                except Exception as e:
                    logger.warning(f"Could not get page count from reader.pages: {e}")
                    try:
                        # Alternative method
                        total_pages = reader.numPages if hasattr(reader, 'numPages') else 0
                    except:
                        total_pages = 0

                logger.debug(f"PDF has {total_pages} pages, requesting page {page_num + 1}")

                # Validate page number with extra safety checks
                if page_num < 0:
                    return False, f"Invalid page number: {page_num + 1} (negative)"
                if total_pages > 0 and page_num >= total_pages:
                    return False, f"Page {page_num + 1} does not exist (PDF has {total_pages} pages)"

                # Additional safety checks before accessing page
                try:
                    if not reader.pages:
                        return False, "PDF has no accessible pages"

                    if page_num >= len(reader.pages):
                        return False, f"Page index {page_num} out of range (pages list length: {len(reader.pages)})"

                    # Multiple extraction attempts
                    page = reader.pages[page_num]

                    # Try different extraction methods
                    text = ""
                    try:
                        text = page.extract_text()
                    except Exception as e:
                        logger.warning(f"extract_text() failed for page {page_num + 1}: {e}")
                        try:
                            # Alternative method
                            text = page.extractText() if hasattr(page, 'extractText') else ""
                        except Exception as e2:
                            logger.warning(f"extractText() also failed for page {page_num + 1}: {e2}")
                            text = ""

                    logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")
                    return True, text.strip()

                except (IndexError, KeyError, AttributeError, TypeError) as e:
                    logger.error(f"PyPDF2 page access error for page {page_num + 1}: {e}")
                    return False, f"Cannot access page {page_num + 1}: {e}"

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for page {page_num + 1}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"PyPDF2 extraction error: {e}"

    def extract_page(self, pdf_path: str, page_num: int, method: str = 'pymupdf') -> Tuple[bool, str]:
        """Extract text from a specific page with fallback methods"""
        try:
            logger.debug(f"Extracting page {page_num + 1} using method: {method}")

            # Try primary method first
            if method == 'pymupdf':
                success, result = self.extract_page_pymupdf(pdf_path, page_num)
                if success and result.strip():
                    return True, result

                # Fallback to PyPDF2
                logger.info(f"PyMuPDF failed for page {page_num + 1}, trying PyPDF2...")
                success, result = self.extract_page_pypdf2(pdf_path, page_num)
                return success, result
            else:
                success, result = self.extract_page_pypdf2(pdf_path, page_num)
                if success and result.strip():
                    return True, result

                # Fallback to PyMuPDF
                logger.info(f"PyPDF2 failed for page {page_num + 1}, trying PyMuPDF...")
                success, result = self.extract_page_pymupdf(pdf_path, page_num)
                return success, result

        except Exception as e:
            logger.error(f"Extract page failed for page {page_num + 1}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Failed to extract text from page {page_num + 1}: {e}"

    def clean_page_text(self, text: str) -> str:
        """Basic cleaning for page text with error handling"""
        try:
            if not text:
                return ""

            logger.debug(f"Cleaning text: {len(text)} chars")

            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Fix broken words at line breaks
            text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
            # Remove page numbers and headers/footers (basic patterns)
            text = re.sub(r'^(Page \d+|\d+\s*$)', '', text, flags=re.MULTILINE)
            # Expand common abbreviations for better TTS
            text = text.replace('Dr.', 'Doctor').replace('Prof.', 'Professor')
            text = text.replace('vs.', 'versus').replace('etc.', 'et cetera')

            cleaned = text.strip()
            logger.debug(f"Cleaned text: {len(cleaned)} chars")
            return cleaned

        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")

    def diagnose_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Comprehensive PDF diagnosis for debugging"""
        diagnosis = {
            "file_exists": False,
            "file_size": 0,
            "is_readable": False,
            "pymupdf_info": {},
            "pypdf2_info": {},
            "recommended_method": None,
            "issues": []
        }

        try:
            # Basic file checks
            diagnosis["file_exists"] = os.path.exists(pdf_path)
            if diagnosis["file_exists"]:
                diagnosis["file_size"] = os.path.getsize(pdf_path)
                diagnosis["is_readable"] = os.access(pdf_path, os.R_OK)
            else:
                diagnosis["issues"].append("File does not exist")
                return diagnosis

            # PyMuPDF diagnosis
            try:
                doc = fitz.open(pdf_path)
                diagnosis["pymupdf_info"] = {
                    "can_open": True,
                    "page_count": doc.page_count,
                    "is_encrypted": doc.is_encrypted,
                    "needs_pass": doc.needs_pass,
                    "metadata": doc.metadata,
                    "error": None
                }

                # Test first page extraction
                if doc.page_count > 0:
                    try:
                        page = doc.load_page(0)
                        test_text = page.get_text()
                        diagnosis["pymupdf_info"]["first_page_text_length"] = len(test_text)
                        diagnosis["pymupdf_info"]["can_extract_text"] = True
                    except Exception as e:
                        diagnosis["pymupdf_info"]["can_extract_text"] = False
                        diagnosis["pymupdf_info"]["extraction_error"] = str(e)

                doc.close()
            except Exception as e:
                diagnosis["pymupdf_info"] = {
                    "can_open": False,
                    "error": str(e)
                }
                diagnosis["issues"].append(f"PyMuPDF error: {e}")

            # PyPDF2 diagnosis
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    diagnosis["pypdf2_info"] = {
                        "can_open": True,
                        "is_encrypted": reader.is_encrypted,
                        "error": None
                    }

                    # Try to get page count
                    try:
                        page_count = len(reader.pages)
                        diagnosis["pypdf2_info"]["page_count"] = page_count
                        diagnosis["pypdf2_info"]["has_pages"] = page_count > 0
                    except Exception as e:
                        diagnosis["pypdf2_info"]["page_count"] = 0
                        diagnosis["pypdf2_info"]["page_count_error"] = str(e)

                    # Test first page extraction
                    if diagnosis["pypdf2_info"].get("page_count", 0) > 0:
                        try:
                            page = reader.pages[0]
                            test_text = page.extract_text()
                            diagnosis["pypdf2_info"]["first_page_text_length"] = len(test_text)
                            diagnosis["pypdf2_info"]["can_extract_text"] = True
                        except Exception as e:
                            diagnosis["pypdf2_info"]["can_extract_text"] = False
                            diagnosis["pypdf2_info"]["extraction_error"] = str(e)

            except Exception as e:
                diagnosis["pypdf2_info"] = {
                    "can_open": False,
                    "error": str(e)
                }
                diagnosis["issues"].append(f"PyPDF2 error: {e}")

            # Determine recommended method
            pymupdf_ok = diagnosis["pymupdf_info"].get("can_open", False) and \
                         diagnosis["pymupdf_info"].get("page_count", 0) > 0
            pypdf2_ok = diagnosis["pypdf2_info"].get("can_open", False) and \
                        diagnosis["pypdf2_info"].get("page_count", 0) > 0

            if pymupdf_ok and pypdf2_ok:
                diagnosis["recommended_method"] = "both_available"
            elif pymupdf_ok:
                diagnosis["recommended_method"] = "pymupdf"
            elif pypdf2_ok:
                diagnosis["recommended_method"] = "pypdf2"
            else:
                diagnosis["recommended_method"] = "none"
                diagnosis["issues"].append("Neither PyMuPDF nor PyPDF2 can process this file")

        except Exception as e:
            diagnosis["issues"].append(f"Diagnosis failed: {e}")
            logger.error(f"PDF diagnosis failed: {e}")

        return diagnosis


# --- Voice Manager ---
class VoiceManager:
    """Manage voice samples and configurations"""

    def __init__(self):
        self.voice_config_file = "voices.json"
        self.voices = self.load_voices()

    def load_voices(self) -> dict:
        """Load voice configuration"""
        try:
            if os.path.exists(self.voice_config_file):
                with open(self.voice_config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load voice config: {e}")

        # Default configuration
        return {
            "standard": {
                "name": "Standard Voice",
                "type": "standard",
                "enabled": True
            }
        }

    def save_voices(self):
        """Save voice configuration"""
        try:
            with open(self.voice_config_file, 'w') as f:
                json.dump(self.voices, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save voice config: {e}")

    def get_available_voices(self) -> list[str]:
        """Get list of available voices"""
        try:
            voices = ["🔊 Standard Voice"]

            # Add custom voices
            for voice_id, voice_info in self.voices.items():
                if voice_id != "standard" and voice_info.get("enabled", True):
                    name = voice_info.get("name", voice_id)
                    voices.append(f"🎭 {name}")

            return voices
        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return ["🔊 Standard Voice"]

    def get_voice_id(self, display_name: str) -> str:
        """Get voice ID from display name"""
        try:
            if "Standard Voice" in display_name:
                return "standard"

            # Extract name from display format
            if display_name.startswith("🎭 "):
                name = display_name[3:]
                for voice_id, voice_info in self.voices.items():
                    if voice_info.get("name") == name:
                        return voice_id
        except Exception as e:
            logger.error(f"Failed to get voice ID: {e}")

        return "standard"

    def add_voice(self, voice_id: str, name: str, audio_path: str) -> bool:
        """Add new custom voice"""
        try:
            # Create voice directory
            voice_dir = Path(config.VOICE_SAMPLES_DIR) / voice_id
            voice_dir.mkdir(exist_ok=True)

            # Save reference audio
            reference_path = voice_dir / "reference.wav"

            # Convert audio to WAV if needed
            waveform, sample_rate = torchaudio.load(audio_path)
            torchaudio.save(str(reference_path), waveform, sample_rate)

            # Update configuration
            self.voices[voice_id] = {
                "name": name,
                "type": "custom",
                "enabled": True,
                "created_at": datetime.now().isoformat()
            }

            self.save_voices()
            logger.info(f"Added voice: {name} ({voice_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add voice: {e}")
            return False


# --- TTS Generator ---
class TTSGenerator:
    """Generate speech from text"""

    def __init__(self):
        self.voice_model = None
        self.model_loaded = False

    def load_model(self):
        """Load TTS model (lazy loading)"""
        if not self.model_loaded:
            try:
                logger.info("Loading TTS model...")
                self.voice_model = TTS(
                    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                    progress_bar=False
                )
                self.model_loaded = True
                logger.info("TTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load TTS model: {e}")
                raise

    def generate_audio_file(self, text: str, voice_id: str, output_path: str) -> bool:
        """Generate audio file from text"""
        if not text.strip():
            return False

        try:
            self.load_model()

            # Clean text for TTS
            clean_text = self.clean_text_for_tts(text)

            if voice_id == "standard":
                # Use default voice
                self.voice_model.tts_to_file(
                    text=clean_text,
                    file_path=output_path,
                    language="en"
                )
            else:
                # Use custom voice
                reference_audio = os.path.join(config.VOICE_SAMPLES_DIR, voice_id, "reference.wav")
                if not os.path.exists(reference_audio):
                    logger.error(f"Reference audio not found for voice: {voice_id}")
                    return False

                self.voice_model.tts_to_file(
                    text=clean_text,
                    file_path=output_path,
                    speaker_wav=reference_audio,
                    language="en"
                )

            return True

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return False

    def clean_text_for_tts(self, text: str) -> str:
        """Clean text for optimal TTS output"""
        try:
            if not text:
                return ""

            # Remove problematic characters
            text = re.sub(r'[^\w\s.,!?;:\-\'"\(\)]', ' ', text)

            # Fix spacing
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.!?,:;])', r'\1', text)

            # Convert numbers to words (basic)
            def number_to_words(match):
                try:
                    num = int(match.group(0))
                    if 0 <= num <= 9999:
                        return num2words(num)
                except:
                    pass
                return match.group(0)

            text = re.sub(r'\b\d+\b', number_to_words, text)

            # Ensure proper sentence ending
            text = text.strip()
            if text and text[-1] not in '.!?':
                text += '.'

            return text
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text if text else ""


# --- Global Instances ---
ollama_processor = OllamaProcessor()
pdf_extractor = PDFPageExtractor()
voice_manager = VoiceManager()
tts_generator = TTSGenerator()


# --- Main Processing Function with Enhanced Debug ---
def process_pdf_pages(pdf_file, start_page, end_page, use_ollama, progress=gr.Progress()):
    """Process PDF pages with comprehensive error handling and debugging"""

    logger.info("=== Starting PDF Processing ===")
    logger.debug(
        f"Input parameters: pdf_file={pdf_file}, start_page={start_page}, end_page={end_page}, use_ollama={use_ollama}")

    try:
        if not pdf_file:
            return "", "❌ Please upload a PDF file!"

        pdf_path = pdf_file.name
        logger.info(f"Processing PDF: {pdf_path}")

        # Validate PDF with comprehensive diagnosis
        logger.info("Running comprehensive PDF diagnosis...")
        diagnosis = pdf_extractor.diagnose_pdf(pdf_path)
        logger.info(f"PDF diagnosis: {diagnosis}")

        if not diagnosis["file_exists"]:
            return "", "❌ PDF file does not exist"

        if diagnosis["recommended_method"] == "none":
            issues = "; ".join(diagnosis["issues"])
            return "", f"❌ Cannot process PDF: {issues}"

        # Use diagnosis info for better processing
        if diagnosis["recommended_method"] == "pymupdf":
            preferred_method = "pymupdf"
            logger.info("Using PyMuPDF based on diagnosis")
        elif diagnosis["recommended_method"] == "pypdf2":
            preferred_method = "pypdf2"
            logger.info("Using PyPDF2 based on diagnosis")
        else:
            preferred_method = "pymupdf"  # Default
            logger.info("Using PyMuPDF as default method")

        # Get page count from diagnosis
        pymupdf_pages = diagnosis.get("pymupdf_info", {}).get("page_count", 0)
        pypdf2_pages = diagnosis.get("pypdf2_info", {}).get("page_count", 0)
        total_pages = max(pymupdf_pages, pypdf2_pages)

        if total_pages == 0:
            logger.error("PDF has no readable pages according to diagnosis")
            return "", "❌ Could not read PDF pages - diagnosis shows 0 pages"

        logger.info(f"PDF has {total_pages} pages")

        # Validate and fix page range - be very explicit
        try:
            start_page = int(start_page)
            end_page = int(end_page)
            logger.debug(f"Original page range: {start_page}-{end_page}")

            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))
            logger.info(f"Validated page range: {start_page}-{end_page}")

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid page range values: {e}")
            return "", f"❌ Invalid page range: {e}"

        # Check Ollama if needed
        if use_ollama:
            is_connected, conn_msg = ollama_processor.check_connection()
            if not is_connected:
                logger.warning(f"Ollama not available: {conn_msg}")
                use_ollama = False  # Disable Ollama but continue

        # Create page list - this is where the error might occur
        try:
            pages_to_process = list(range(start_page - 1, end_page))  # Convert to 0-based
            logger.info(f"Pages to process (0-based): {pages_to_process}")

            if not pages_to_process:
                return "", "❌ No pages to process"

        except Exception as e:
            logger.error(f"Error creating page list: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "", f"❌ Error creating page list: {e}"

        # Process pages with maximum error protection
        processed_texts = []
        failed_pages = []

        for i, page_num in enumerate(pages_to_process):
            try:
                logger.info(f"=== Processing page {page_num + 1} ({i + 1}/{len(pages_to_process)}) ===")

                if progress:
                    progress((i + 1) / len(pages_to_process), f"Processing page {page_num + 1}...")

                # Extra validation
                if page_num < 0 or page_num >= total_pages:
                    error_msg = f"Page number {page_num + 1} out of range (1-{total_pages})"
                    logger.error(error_msg)
                    failed_pages.append((page_num + 1, error_msg))
                    continue

                # Extract page text with comprehensive error handling
                try:
                    success, page_text = pdf_extractor.extract_page(pdf_path, page_num, method=preferred_method)
                    logger.debug(
                        f"Page extraction result: success={success}, text_length={len(page_text) if isinstance(page_text, str) else 'N/A'}")

                except Exception as e:
                    error_msg = f"Extraction exception: {e}"
                    logger.error(f"Page {page_num + 1} extraction failed: {error_msg}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    failed_pages.append((page_num + 1, error_msg))
                    continue

                if not success:
                    logger.warning(f"Failed to extract page {page_num + 1}: {page_text}")
                    failed_pages.append((page_num + 1, page_text or "Unknown extraction error"))
                    continue

                if not page_text or not page_text.strip():
                    logger.info(f"Page {page_num + 1} is empty, skipping...")
                    continue

                # Clean basic formatting
                try:
                    cleaned_text = pdf_extractor.clean_page_text(page_text)
                    logger.debug(f"Text cleaned successfully: {len(cleaned_text)} chars")
                except Exception as e:
                    logger.error(f"Text cleaning failed for page {page_num + 1}: {e}")
                    cleaned_text = page_text  # Use original if cleaning fails

                # Process with Ollama if enabled
                if use_ollama and cleaned_text.strip():
                    try:
                        logger.debug(f"Sending page {page_num + 1} to Ollama...")
                        success, tts_text = ollama_processor.process_page_for_tts(cleaned_text)
                        if success:
                            processed_texts.append(f"--- Page {page_num + 1} ---\n{tts_text}\n")
                            logger.info(f"Page {page_num + 1} processed with Ollama successfully")
                        else:
                            logger.warning(f"Ollama processing failed for page {page_num + 1}: {tts_text}")
                            processed_texts.append(f"--- Page {page_num + 1} ---\n{cleaned_text}\n")
                    except Exception as e:
                        logger.error(f"Ollama error for page {page_num + 1}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        processed_texts.append(f"--- Page {page_num + 1} ---\n{cleaned_text}\n")
                else:
                    processed_texts.append(f"--- Page {page_num + 1} ---\n{cleaned_text}\n")
                    logger.info(f"Page {page_num + 1} processed without Ollama")

            except Exception as e:
                error_msg = f"Processing error: {e}"
                logger.error(f"Error processing page {page_num + 1}: {error_msg}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                failed_pages.append((page_num + 1, error_msg))
                continue

        logger.info(f"Processing complete: {len(processed_texts)} successful, {len(failed_pages)} failed")

        # Generate results
        if not processed_texts:
            error_msg = "❌ No text extracted from selected pages"
            if failed_pages:
                error_msg += f"\n\nFailed pages: {', '.join([str(p[0]) for p in failed_pages])}"
                error_msg += f"\nErrors: {'; '.join([p[1] for p in failed_pages[:3]])}"
            logger.error(error_msg)
            return "", error_msg

        # Combine all processed text
        try:
            final_text = "\n".join(processed_texts)
            logger.info(f"Final text length: {len(final_text)} characters")
        except Exception as e:
            logger.error(f"Error combining texts: {e}")
            return "", f"❌ Error combining processed texts: {e}"

        # Save processed text
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pages_{start_page}-{end_page}_{timestamp}.txt"
            output_path = Path(config.PROCESSED_PAGES_DIR) / filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            logger.info(f"Saved processed text to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save processed text: {e}")

        # Build status message
        success_count = len(processed_texts)
        total_requested = len(pages_to_process)

        status = f"""✅ Processing completed!
📄 Pages processed: {success_count}/{total_requested} (pages {start_page}-{end_page})
🤖 Ollama enhancement: {'Yes' if use_ollama else 'No'}
📝 Total text length: {len(final_text):,} characters
💾 Saved as: {filename}"""

        if failed_pages:
            status += f"\n\n⚠️ Failed pages: {', '.join([str(p[0]) for p in failed_pages])}"

        logger.info("=== PDF Processing Complete ===")
        return final_text, status

    except Exception as e:
        logger.error(f"PDF processing failed with exception: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return "", f"❌ Processing failed: {str(e)}\n\nCheck the logs (pdf_tts_debug.log) for detailed error information."


def check_ollama_connection():
    """Check Ollama connection status"""
    is_connected, message = ollama_processor.check_connection()
    return message


def generate_audio_from_text(text, voice_display, progress=gr.Progress()):
    """Generate audio from processed text"""
    try:
        logger.info("=== Starting Audio Generation ===")

        if not text.strip():
            return None, [], "❌ No text to convert to audio!"

        if not voice_display:
            return None, [], "❌ Please select a voice!"

        voice_id = voice_manager.get_voice_id(voice_display)
        logger.info(f"Using voice: {voice_display} (ID: {voice_id})")

        # Split text by pages with error handling
        try:
            page_sections = re.split(r'--- Page \d+ ---', text)
            page_sections = [section.strip() for section in page_sections if section.strip()]

            if not page_sections:
                # If no page markers, treat as single text
                page_sections = [text]

            logger.info(f"Split text into {len(page_sections)} sections")
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            page_sections = [text]

        audio_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, section in enumerate(page_sections):
            try:
                if progress:
                    progress((i + 1) / len(page_sections), f"Generating audio {i + 1}/{len(page_sections)}...")

                # Generate filename
                output_filename = f"page_{i + 1:03d}_{timestamp}.wav"
                output_path = os.path.join(config.GENERATED_AUDIO_DIR, output_filename)

                # Generate audio
                success = tts_generator.generate_audio_file(section, voice_id, output_path)
                if success:
                    audio_files.append(output_path)
                    logger.info(f"Generated audio: {output_filename}")
                else:
                    logger.warning(f"Failed to generate audio for section {i + 1}")
            except Exception as e:
                logger.error(f"Error generating audio for section {i + 1}: {e}")

        if audio_files:
            status = f"""✅ Audio generation completed!
🎵 Generated: {len(audio_files)} files
🎭 Voice: {voice_display}
📊 Sections: {len(page_sections)}
🔊 First file duration: ~{len(page_sections[0]) // 150} seconds (estimated)"""

            logger.info("=== Audio Generation Complete ===")
            return audio_files[0], audio_files, status
        else:
            return None, [], "❌ Failed to generate any audio files!"

    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, [], f"❌ Audio generation failed: {str(e)}"


def upload_voice_sample(audio_file, voice_name):
    """Upload and register new voice sample"""
    if not audio_file or not voice_name.strip():
        return "❌ Please provide both audio file and voice name!", gr.update()

    try:
        # Validate voice name
        voice_name = voice_name.strip()
        if len(voice_name) < 2 or len(voice_name) > 30:
            return "❌ Voice name must be 2-30 characters!", gr.update()

        # Generate voice ID
        voice_id = re.sub(r'[^\w]', '_', voice_name.lower())

        # Check file size
        file_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
        if file_size > 20:
            return f"❌ Audio file too large: {file_size:.1f}MB (max 20MB)!", gr.update()

        # Validate audio
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            duration = waveform.shape[1] / sample_rate

            if duration < 3:
                return f"❌ Audio too short: {duration:.1f}s (minimum 3s)!", gr.update()
            if duration > 60:
                return f"❌ Audio too long: {duration:.1f}s (maximum 60s)!", gr.update()
        except Exception as e:
            return f"❌ Invalid audio file: {e}", gr.update()

        # Add voice
        success = voice_manager.add_voice(voice_id, voice_name, audio_file)

        if success:
            # Update voice list
            new_voices = voice_manager.get_available_voices()
            updated_dropdown = gr.Dropdown(choices=new_voices, value=f"🎭 {voice_name}")

            status = f"""✅ Voice '{voice_name}' uploaded successfully!
🎵 Duration: {duration:.1f} seconds
📊 Sample rate: {sample_rate} Hz
💾 Voice ID: {voice_id}"""

            return status, updated_dropdown
        else:
            return "❌ Failed to save voice sample!", gr.update()

    except Exception as e:
        logger.error(f"Voice upload failed: {e}")
        return f"❌ Upload failed: {str(e)}", gr.update()


def refresh_voices():
    """Refresh voice dropdown"""
    voices = voice_manager.get_available_voices()
    return gr.Dropdown(choices=voices, value=voices[0] if voices else None)


def update_page_range(pdf_file):
    """Update page range slider based on PDF"""
    if not pdf_file:
        return gr.update(maximum=1, value=1), gr.update(maximum=1, value=1)

    try:
        logger.debug(f"Updating page range for: {pdf_file.name}")
        total_pages = pdf_extractor.get_page_count(pdf_file.name)
        logger.debug(f"Total pages: {total_pages}")

        if total_pages > 0:
            return (
                gr.update(maximum=total_pages, value=1),
                gr.update(maximum=total_pages, value=min(total_pages, 5))
            )
    except Exception as e:
        logger.error(f"Error updating page range: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    return gr.update(maximum=1, value=1), gr.update(maximum=1, value=1)


def diagnose_pdf_file(pdf_file):
    """Diagnose PDF file for debugging"""
    if not pdf_file:
        return "❌ No PDF file provided"

    try:
        diagnosis = pdf_extractor.diagnose_pdf(pdf_file.name)

        report = f"""📄 PDF Diagnosis Report for: {os.path.basename(pdf_file.name)}

📁 **File Info:**
- Exists: {'✅' if diagnosis['file_exists'] else '❌'}
- Size: {diagnosis['file_size'] / 1024 / 1024:.2f} MB
- Readable: {'✅' if diagnosis['is_readable'] else '❌'}

🔧 **PyMuPDF Status:**
- Can Open: {'✅' if diagnosis['pymupdf_info'].get('can_open', False) else '❌'}
- Page Count: {diagnosis['pymupdf_info'].get('page_count', 'N/A')}
- Encrypted: {'⚠️' if diagnosis['pymupdf_info'].get('is_encrypted', False) else '✅'}
- Can Extract Text: {'✅' if diagnosis['pymupdf_info'].get('can_extract_text', False) else '❌'}

🔧 **PyPDF2 Status:**
- Can Open: {'✅' if diagnosis['pypdf2_info'].get('can_open', False) else '❌'}
- Page Count: {diagnosis['pypdf2_info'].get('page_count', 'N/A')}
- Encrypted: {'⚠️' if diagnosis['pypdf2_info'].get('is_encrypted', False) else '✅'}
- Can Extract Text: {'✅' if diagnosis['pypdf2_info'].get('can_extract_text', False) else '❌'}

💡 **Recommendation:** {diagnosis['recommended_method']}

⚠️ **Issues Found:**
{chr(10).join(f"- {issue}" for issue in diagnosis['issues']) if diagnosis['issues'] else 'None'}
"""
        return report

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        return f"❌ Diagnosis failed: {str(e)}"
    """Count text statistics"""
    try:
        if not text:
            return 0, 0, "0 minutes"

        words = len(text.split())
        chars = len(text)
        # Estimate 150 words per minute reading speed
        minutes = max(1, words // 150)

        if minutes == 1:
            time_str = "1 minute"
        elif minutes < 60:
            time_str = f"{minutes} minutes"
        else:
            hours = minutes // 60
            remaining_mins = minutes % 60
            time_str = f"{hours}h {remaining_mins}m"

        return words, chars, time_str
    except Exception as e:
        logger.error(f"Error counting text stats: {e}")
        return 0, 0, "Error"


# --- Create Gradio Interface ---
def create_interface():
    """Create simplified Gradio interface"""

    with gr.Blocks(
            title="📄 DEBUG PDF to TTS with AI Processing",
            theme=gr.themes.Soft(),
            css="""
        .container { max-width: 1200px; margin: auto; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 10px; border-radius: 5px; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 10px; border-radius: 5px; }
        .info { background-color: #cce7ff; border: 1px solid #99d6ff; color: #004085; padding: 10px; border-radius: 5px; }
        """
    ) as demo:

        gr.Markdown("# 📄 DEBUG PDF to TTS with AI Processing")
        gr.Markdown("**Enhanced debug version with comprehensive error logging**")
        gr.Markdown("**Check `pdf_tts_debug.log` for detailed debugging information**")

        with gr.Tab("📄 PDF Processing"):
            gr.Markdown("## Step 1: Upload PDF and Select Pages")

            with gr.Row():
                with gr.Column(scale=2):
                    pdf_input = gr.File(
                        label="📁 Upload PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Page Selection")
                    start_page = gr.Slider(
                        minimum=1,
                        maximum=1,
                        value=1,
                        step=1,
                        label="📄 Start Page"
                    )

                    end_page = gr.Slider(
                        minimum=1,
                        maximum=1,
                        value=1,
                        step=1,
                        label="📄 End Page"
                    )

                    use_ollama = gr.Checkbox(
                        label="🤖 Enhance with AI",
                        value=False,  # Disable by default for debugging
                        info="Optimize text for speech synthesis"
                    )

            # Debug info
            with gr.Row():
                gr.Markdown("### 🐛 Debug Information")
                debug_info = gr.Textbox(
                    label="Debug Info",
                    value="Upload PDF to see debug information...",
                    interactive=False,
                    lines=3
                )

            # Ollama status
            with gr.Row():
                ollama_status = gr.Textbox(
                    label="🤖 Ollama Status",
                    value="Click 'Check Connection' to verify",
                    interactive=False,
                    lines=2
                )
                check_ollama_btn = gr.Button("🔗 Check Connection", size="sm")

            process_pages_btn = gr.Button("📝 Process Selected Pages", variant="primary", size="lg")

            processing_status = gr.Textbox(
                label="📊 Processing Status",
                lines=8,
                interactive=False,
                placeholder="Upload PDF and click 'Process Selected Pages' to begin..."
            )

            # Processed text area (editable)
            processed_text = gr.Textbox(
                label="✏️ Processed Text (Editable)",
                lines=20,
                placeholder="Processed text will appear here. You can edit it before generating audio...",
                info="Edit the text as needed before converting to speech"
            )

        with gr.Tab("✏️ Text Editor"):
            gr.Markdown("## Step 2: Review and Edit Your Text")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Text Statistics")
                    word_count = gr.Number(label="📝 Words", value=0, interactive=False)
                    char_count = gr.Number(label="🔤 Characters", value=0, interactive=False)
                    reading_time = gr.Textbox(label="⏰ Reading Time", value="0 minutes", interactive=False)

                    gr.Markdown("### 🛠️ Quick Actions")
                    clear_text_btn = gr.Button("🗑️ Clear Text", size="sm", variant="stop")
                    load_file_btn = gr.File(
                        label="📂 Load Text File",
                        file_types=[".txt"],
                        type="filepath",
                        scale=0
                    )

                with gr.Column(scale=3):
                    # Main text editor
                    text_editor = gr.Textbox(
                        label="📝 Text Editor",
                        lines=25,
                        placeholder="Your processed text will appear here for editing...",
                        info="Make any edits needed before generating audio"
                    )

            with gr.Row():
                with gr.Column():
                    find_text = gr.Textbox(
                        label="🔍 Find Text",
                        placeholder="Enter text to find...",
                        max_lines=1
                    )

                with gr.Column():
                    replace_text = gr.Textbox(
                        label="🔄 Replace With",
                        placeholder="Enter replacement text...",
                        max_lines=1
                    )

                with gr.Column():
                    find_btn = gr.Button("🔍 Find", size="sm")
                    replace_btn = gr.Button("🔄 Replace All", size="sm", variant="secondary")

            find_replace_status = gr.Textbox(
                label="🔍 Search Results",
                lines=2,
                interactive=False
            )

        with gr.Tab("🎙️ Audio Generation"):
            gr.Markdown("## Step 3: Generate High-Quality Speech")

            with gr.Row():
                with gr.Column(scale=2):
                    voice_selector = gr.Dropdown(
                        label="🎭 Select Voice",
                        choices=voice_manager.get_available_voices(),
                        value=voice_manager.get_available_voices()[0] if voice_manager.get_available_voices() else None,
                        info="Choose from available voices"
                    )

                with gr.Column(scale=1):
                    refresh_voices_btn = gr.Button("🔄 Refresh Voices", size="sm")

            # Text preview for audio generation
            audio_text_preview = gr.Textbox(
                label="📝 Text for Audio Generation",
                lines=12,
                interactive=False,
                placeholder="Click 'Load Text for Audio' to see the text that will be converted..."
            )

            load_text_btn = gr.Button("📋 Load Text for Audio", variant="secondary")

            generate_audio_btn = gr.Button("🔊 Generate Audio", variant="primary", size="lg")

            # Audio outputs
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
                label="📊 Audio Generation Status",
                lines=6,
                interactive=False,
                placeholder="Load text and click 'Generate Audio' to begin..."
            )

        with gr.Tab("🎭 Voice Training"):
            gr.Markdown("## Step 4: Train Custom Voices (Optional)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 Upload Voice Sample")

                    voice_upload = gr.File(
                        label="🎵 Audio Sample",
                        file_types=config.SUPPORTED_AUDIO_FORMATS,
                        type="filepath"
                    )

                    voice_name = gr.Textbox(
                        label="🏷️ Voice Name",
                        placeholder="e.g., 'John's Voice', 'Narrator Voice'",
                        max_lines=1
                    )

                    upload_voice_btn = gr.Button("⬆️ Train Voice", variant="primary")

                    upload_status = gr.Textbox(
                        label="📊 Training Status",
                        lines=6,
                        interactive=False
                    )

                with gr.Column():
                    gr.Markdown("### 📋 Voice Sample Requirements")
                    gr.Markdown("""
                    **For best results:**

                    ✅ **Duration**: 5-30 seconds of clear speech  
                    ✅ **Quality**: No background noise or music  
                    ✅ **Speaker**: Single person speaking naturally  
                    ✅ **Content**: Complete sentences work best  
                    ✅ **Format**: WAV, MP3, FLAC, or OGG  
                    ✅ **Size**: Under 20MB  

                    **Tips:**
                    - Record in a quiet room
                    - Speak at normal conversational pace
                    - Include varied intonation
                    - Avoid breathing sounds or pauses
                    - Use good quality microphone if available
                    """)

        with gr.Tab("🐛 Debug Logs"):
            gr.Markdown("## Debug Information")
            gr.Markdown("**Real-time debug information and logs**")

            with gr.Row():
                log_refresh_btn = gr.Button("🔄 Refresh Logs", size="sm")
                clear_logs_btn = gr.Button("🗑️ Clear Logs", size="sm", variant="stop")

            debug_logs = gr.Textbox(
                label="Debug Logs",
                lines=30,
                interactive=False,
                placeholder="Debug logs will appear here..."
            )

        # Event Handlers

        # PDF upload updates page sliders and debug info
        def update_debug_info(pdf_file):
            start_slider, end_slider = update_page_range(pdf_file)

            debug_text = "No PDF uploaded"
            if pdf_file:
                try:
                    total_pages = pdf_extractor.get_page_count(pdf_file.name)
                    file_size = os.path.getsize(pdf_file.name) / (1024 * 1024)
                    debug_text = f"""📄 File: {os.path.basename(pdf_file.name)}
📊 Size: {file_size:.2f} MB
📖 Pages: {total_pages}
✅ Status: Ready for processing"""
                except Exception as e:
                    debug_text = f"❌ Error reading PDF: {e}"

            return start_slider, end_slider, debug_text

        pdf_input.change(
            fn=update_debug_info,
            inputs=[pdf_input],
            outputs=[start_page, end_page, debug_info]
        )

        # Check Ollama connection
        check_ollama_btn.click(
            fn=check_ollama_connection,
            outputs=[ollama_status]
        )

        # Process PDF pages
        process_pages_btn.click(
            fn=process_pdf_pages,
            inputs=[pdf_input, start_page, end_page, use_ollama],
            outputs=[processed_text, processing_status]
        )

        # Sync processed text to editor
        processed_text.change(
            fn=lambda text: text,
            inputs=[processed_text],
            outputs=[text_editor]
        )

        # Update text statistics
        text_editor.change(
            fn=count_text_stats,
            inputs=[text_editor],
            outputs=[word_count, char_count, reading_time]
        )

        # Text editing functions
        def clear_text():
            return "", "Text cleared"

        def load_text_file(file_path):
            if not file_path:
                return gr.update(), "No file selected"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, f"Loaded {len(content):,} characters from file"
            except Exception as e:
                return gr.update(), f"Error loading file: {e}"

        def find_text_function(text, find_term):
            if not text or not find_term:
                return text, "Please enter text to find"

            count = len(re.findall(re.escape(find_term), text, re.IGNORECASE))
            if count > 0:
                return text, f"Found {count} matches for '{find_term}'"
            else:
                return text, f"No matches found for '{find_term}'"

        def replace_text_function(text, find_term, replace_term):
            if not text or not find_term:
                return text, "Please enter text to find and replace"

            original_count = len(re.findall(re.escape(find_term), text, re.IGNORECASE))
            if original_count == 0:
                return text, f"No matches found for '{find_term}'"

            new_text = re.sub(re.escape(find_term), replace_term, text, flags=re.IGNORECASE)
            return new_text, f"Replaced {original_count} occurrences of '{find_term}' with '{replace_term}'"

        def refresh_debug_logs():
            """Load recent log entries"""
            try:
                if os.path.exists('pdf_tts_debug.log'):
                    with open('pdf_tts_debug.log', 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Get last 100 lines
                        recent_logs = ''.join(lines[-100:])
                        return recent_logs
                else:
                    return "No log file found yet..."
            except Exception as e:
                return f"Error reading logs: {e}"

        def clear_debug_logs():
            """Clear the debug log file"""
            try:
                with open('pdf_tts_debug.log', 'w') as f:
                    f.write("")
                return "Debug logs cleared"
            except Exception as e:
                return f"Error clearing logs: {e}"

        clear_text_btn.click(
            fn=clear_text,
            outputs=[text_editor, find_replace_status]
        )

        load_file_btn.change(
            fn=load_text_file,
            inputs=[load_file_btn],
            outputs=[text_editor, find_replace_status]
        )

        find_btn.click(
            fn=find_text_function,
            inputs=[text_editor, find_text],
            outputs=[text_editor, find_replace_status]
        )

        replace_btn.click(
            fn=replace_text_function,
            inputs=[text_editor, find_text, replace_text],
            outputs=[text_editor, find_replace_status]
        )

        # Debug log functions
        log_refresh_btn.click(
            fn=refresh_debug_logs,
            outputs=[debug_logs]
        )

        clear_logs_btn.click(
            fn=clear_debug_logs,
            outputs=[debug_logs]
        )

        # Load text for audio generation
        load_text_btn.click(
            fn=lambda text: text,
            inputs=[text_editor],
            outputs=[audio_text_preview]
        )

        # Voice management
        refresh_voices_btn.click(
            fn=refresh_voices,
            outputs=[voice_selector]
        )

        upload_voice_btn.click(
            fn=upload_voice_sample,
            inputs=[voice_upload, voice_name],
            outputs=[upload_status, voice_selector]
        )

        # Generate audio
        generate_audio_btn.click(
            fn=generate_audio_from_text,
            inputs=[audio_text_preview, voice_selector],
            outputs=[first_audio, all_audio_files, audio_status]
        )

    return demo


# --- Main Application ---
def main():
    """Main application entry point"""

    # Setup logging
    logger.info("🚀 Starting DEBUG PDF to TTS Converter...")
    logger.info(f"📁 Voice samples: {config.VOICE_SAMPLES_DIR}")
    logger.info(f"🔊 Generated audio: {config.GENERATED_AUDIO_DIR}")
    logger.info(f"📄 Processed pages: {config.PROCESSED_PAGES_DIR}")
    logger.info(f"💾 Cache: {config.CACHE_DIR}")

    # Check dependencies
    required_packages = {
        'PyPDF2': 'PyPDF2',
        'fitz': 'PyMuPDF',
        'torchaudio': 'torchaudio',
        'TTS': 'TTS',
        'num2words': 'num2words',
        'requests': 'requests',
        'gradio': 'gradio'
    }

    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            logger.debug(f"✅ {package_name} imported successfully")
        except ImportError:
            missing.append(package_name)
            logger.error(f"❌ {package_name} not found")

    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return

    # Check Ollama connection
    logger.info("🤖 Checking Ollama connection...")
    is_connected, message = ollama_processor.check_connection()
    if is_connected:
        logger.info(f"✅ {message}")
    else:
        logger.warning(f"⚠️ {message}")
        logger.info("💡 Ollama features will be limited until connection is established")

    # Create and launch interface
    try:
        demo = create_interface()
        logger.info("🌐 Launching web interface...")

        demo.launch(
            server_name="0.0.0.0",
            server_port=1602,
            share=False,
            show_error=True,
            quiet=False
        )

    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        logger.info("🔄 Application shutdown complete")


if __name__ == "__main__":
    main()