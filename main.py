import os
import re
import uuid
import tempfile
import requests
from datetime import datetime
from typing import List, Tuple, Dict
from num2words import num2words
import gradio as gr
import json

# PDF Processing
import PyPDF2
import fitz  # PyMuPDF - alternative PDF reader
from pdfplumber import PDF as PDFPlumber

# Audio
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN
from TTS.api import TTS

# --- Constants ---
OLLAMA_BASE_URL = "http://ollama:11434"
DEFAULT_MODEL = "qwen:1.8b"
VOICE_SAMPLES_DIR = "voice_samples"
GENERATED_AUDIO_DIR = "generated_audio"
PROCESSED_PDFS_DIR = "processed_pdfs"
STANDARD_VOICE_NAME = "standard"
WAV_SUFFIX = ".wav"
VOICE_CONFIG_FILE = "voice_config.json"

# Ensure directories exist
os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)
os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
os.makedirs(PROCESSED_PDFS_DIR, exist_ok=True)

# --- PDF Text Extractor ---
class PDFTextExtractor:
    def __init__(self):
        self.extraction_methods = ['pdfplumber', 'pymupdf', 'pypdf2']

    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (best for complex layouts)"""
        text = ""
        try:
            with PDFPlumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"[WARNING] PDFPlumber extraction failed: {e}")
            raise
        return text

    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (good balance of speed and accuracy)"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n\n"
            doc.close()
        except Exception as e:
            print(f"[WARNING] PyMuPDF extraction failed: {e}")
            raise
        return text

    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"[WARNING] PyPDF2 extraction failed: {e}")
            raise
        return text

    def extract_text(self, pdf_path: str, method: str = 'auto') -> str:
        """Extract text from PDF using specified or auto-selected method"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if method == 'auto':
            # Try methods in order of preference
            for extraction_method in self.extraction_methods:
                try:
                    text = self._extract_by_method(pdf_path, extraction_method)
                    if text.strip():
                        print(f"[SUCCESS] Text extracted using {extraction_method}")
                        return text
                except Exception as e:
                    print(f"[INFO] {extraction_method} failed, trying next method...")
                    continue

            raise Exception("All extraction methods failed")
        else:
            return self._extract_by_method(pdf_path, method)

    def _extract_by_method(self, pdf_path: str, method: str) -> str:
        """Extract text using specific method"""
        if method == 'pdfplumber':
            return self.extract_with_pdfplumber(pdf_path)
        elif method == 'pymupdf':
            return self.extract_with_pymupdf(pdf_path)
        elif method == 'pypdf2':
            return self.extract_with_pypdf2(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

# --- Text Cleaner ---
class TextCleaner:
    def __init__(self):
        self.patterns = {
            # Remove headers and footers (common patterns)
            'headers_footers': r'^(Page \d+|\d+\s*$|Chapter \d+.*$)',
            # Remove figure/table references
            'figure_refs': r'(Figure|Fig\.|Table|Tbl\.)\s*\d+[\w\s]*',
            # Remove citation patterns
            'citations': r'\[\d+\]|\(\w+,?\s*\d{4}\)|\(\w+\s+et\s+al\.,?\s*\d{4}\)',
            # Remove URLs
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            # Remove email addresses
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Remove excessive whitespace
            'whitespace': r'\s{3,}',
            # Remove line breaks in the middle of sentences
            'broken_sentences': r'(?<=[a-z])\n(?=[a-z])',
        }

    def clean_academic_text(self, text: str) -> str:
        """Clean academic/research paper text"""
        # Remove common academic artifacts
        text = re.sub(r'Abstract\s*\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'References\s*\n.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Bibliography\s*\n.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\b(doi|DOI):\s*[\w\./\-]+', '', text)

        return text

    def clean_for_tts(self, text: str, preserve_structure: bool = True) -> str:
        """Clean text specifically for TTS conversion"""
        print("[INFO] Cleaning text for TTS...")

        # Store original length
        original_length = len(text)

        # Basic cleaning
        text = text.strip()

        # Remove or replace problematic characters for TTS
        text = re.sub(r'[^\w\s.,!?;:\-\'"()\n]', ' ', text)

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

        # Convert numbers to words for better TTS
        text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0))), text)

        # Handle abbreviations (expand common ones)
        abbreviations = {
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'vs.': 'versus',
            'etc.': 'et cetera',
            'i.e.': 'that is',
            'e.g.': 'for example',
            'Mr.': 'Mister',
            'Mrs.': 'Misses',
            'Ms.': 'Miss',
        }

        for abbrev, expansion in abbreviations.items():
            text = text.replace(abbrev, expansion)

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
        print(f"[INFO] Text cleaned: {original_length} -> {cleaned_length} characters")

        return text

    def split_into_chunks(self, text: str, max_chars: int = 1000) -> List[str]:
        """Split text into manageable chunks for TTS"""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) <= max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

# --- Voice Management (unchanged from original) ---
class VoiceManager:
    def __init__(self):
        self.voice_config_path = VOICE_CONFIG_FILE
        self.voices = self.load_voice_config()

    def load_voice_config(self) -> Dict:
        """Load voice configuration from JSON file"""
        if os.path.exists(self.voice_config_path):
            try:
                with open(self.voice_config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARNING] Error loading voice config: {e}")

        # Default configuration
        default_config = {
            "standard": {
                "name": "Standard Voice",
                "description": "Default synthetic voice",
                "type": "standard",
                "enabled": True
            }
        }
        self.save_voice_config(default_config)
        return default_config

    def save_voice_config(self, config: Dict):
        """Save voice configuration to JSON file"""
        try:
            with open(self.voice_config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Error saving voice config: {e}")

    def get_available_voices(self) -> List[str]:
        """Get list of available voice IDs"""
        voices = []

        # Add standard voice
        if self.voices.get("standard", {}).get("enabled", True):
            voices.append("standard")

        # Scan for cloned voices
        if os.path.exists(VOICE_SAMPLES_DIR):
            for voice_dir in os.listdir(VOICE_SAMPLES_DIR):
                voice_path = os.path.join(VOICE_SAMPLES_DIR, voice_dir)
                reference_file = os.path.join(voice_path, "reference.wav")

                if os.path.isdir(voice_path) and os.path.exists(reference_file):
                    voice_info = self.voices.get(voice_dir, {})
                    if voice_info.get("enabled", True):
                        voices.append(voice_dir)

        return voices if voices else ["standard"]

    def get_voice_display_names(self) -> List[str]:
        """Get list of voice display names"""
        voices = self.get_available_voices()
        display_names = []

        for voice_id in voices:
            if voice_id == "standard":
                display_names.append("Standard Voice")
            else:
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                display_names.append(f"{name} (Cloned)")

        return display_names

    def get_voice_id_from_display(self, display_name: str) -> str:
        """Get voice ID from display name"""
        if display_name == "Standard Voice":
            return "standard"

        for voice_id in self.get_available_voices():
            if voice_id != "standard":
                voice_info = self.voices.get(voice_id, {})
                name = voice_info.get("name", voice_id)
                if display_name == f"{name} (Cloned)":
                    return voice_id

        return "standard"

    def add_voice(self, voice_id: str, name: str, description: str = ""):
        """Add a new voice to the configuration"""
        self.voices[voice_id] = {
            "name": name,
            "description": description,
            "type": "cloned",
            "enabled": True,
            "created_at": datetime.now().isoformat()
        }
        self.save_voice_config(self.voices)

# --- TTS Generator (enhanced from original) ---
class TTSGenerator:
    def __init__(self):
        self.voice_model = None
        self.tacotron2 = None
        self.hifi_gan = None
        self.voice_manager = VoiceManager()
        os.makedirs(GENERATED_AUDIO_DIR, exist_ok=True)

    def preprocess_text(self, text: str) -> str:
        # This is handled by TextCleaner now, but keep for compatibility
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _generate_filename(self, text: str, speaker_id: str, chunk_idx: int = None, fmt: str = "wav") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = re.sub(r'[^\w\s-]', '', text[:30]).strip().replace(" ", "_") or "audio"
        safe_speaker = re.sub(r'[^\w\s-]', '', speaker_id).replace(' ', '_')

        chunk_suffix = f"_part{chunk_idx:03d}" if chunk_idx is not None else ""
        return os.path.join(GENERATED_AUDIO_DIR, f"{ts}_{safe_speaker}_{safe_text}{chunk_suffix}.{fmt}")

    def run_standard_tts(self, text: str, chunk_idx: int = None) -> str:
        if self.tacotron2 is None or self.hifi_gan is None:
            print("Loading standard TTS models...")
            self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmp_tts")
            self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmp_vocoder")

        mel_outputs, _, _ = self.tacotron2.encode_batch([text])
        waveform = self.hifi_gan.decode_batch(mel_outputs).squeeze().detach().cpu()

        file_path = self._generate_filename(text, STANDARD_VOICE_NAME, chunk_idx)
        torchaudio.save(file_path, waveform.unsqueeze(0), 22050)

        print(f"[DEBUG] ✅ Saved standard audio at: {file_path}")
        return file_path

    def run_voice_clone_tts(self, text: str, speaker_id: str, chunk_idx: int = None) -> str:
        if self.voice_model is None:
            print("Loading voice cloning model...")
            self.voice_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

        reference_audio = os.path.join(VOICE_SAMPLES_DIR, speaker_id, "reference.wav")
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"❌ Reference audio for '{speaker_id}' not found at {reference_audio}")

        path = self._generate_filename(text, speaker_id, chunk_idx)
        self.voice_model.tts_to_file(
            text=text,
            file_path=path,
            speaker_wav=reference_audio,
            language="en",
            temperature=0.9,
            split_sentences=True
        )

        print(f"[DEBUG] ✅ Cloned voice saved at: {path}")
        return path

    def generate_audio_from_chunks(self, text_chunks: List[str], voice_id: str) -> List[str]:
        """Generate audio from text chunks"""
        audio_files = []

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue

            try:
                if voice_id == "standard":
                    audio_path = self.run_standard_tts(chunk, i + 1)
                else:
                    audio_path = self.run_voice_clone_tts(chunk, voice_id, i + 1)
                audio_files.append(audio_path)
            except Exception as e:
                print(f"[ERROR] Failed to generate audio for chunk {i + 1}: {str(e)}")
                continue

        return audio_files

    def generate_audio(self, text: str, voice_id: str) -> str:
        """Generate audio with the specified voice (single file)"""
        clean_text = self.preprocess_text(text)

        if not clean_text.strip():
            raise ValueError("❌ No valid text to convert to speech")

        try:
            if voice_id == "standard":
                return self.run_standard_tts(clean_text)
            else:
                return self.run_voice_clone_tts(clean_text, voice_id)
        except Exception as e:
            print(f"[ERROR] Audio generation failed: {str(e)}")
            raise

# --- Global instances ---
pdf_extractor = PDFTextExtractor()
text_cleaner = TextCleaner()
voice_manager = VoiceManager()
tts = TTSGenerator()

# --- Gradio UI Functions ---
def process_pdf(pdf_file, extraction_method, preserve_structure, chunk_size):
    """Process PDF and extract/clean text"""
    if not pdf_file:
        return "", "Please upload a PDF file!"

    try:
        # Extract text from PDF
        print(f"[INFO] Processing PDF: {pdf_file.name}")
        raw_text = pdf_extractor.extract_text(pdf_file.name, extraction_method)

        if not raw_text.strip():
            return "", "❌ No text could be extracted from the PDF!"

        # Clean text for TTS
        cleaned_text = text_cleaner.clean_for_tts(raw_text, preserve_structure)

        # Save processed text for reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(pdf_file.name))[0]
        output_path = os.path.join(PROCESSED_PDFS_DIR, f"{timestamp}_{filename}_cleaned.txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        status = f"✅ PDF processed successfully!\n📄 Extracted {len(raw_text)} characters\n🧹 Cleaned to {len(cleaned_text)} characters\n💾 Saved to: {output_path}"

        return cleaned_text, status

    except Exception as e:
        return "", f"❌ Error processing PDF: {str(e)}"

def get_available_voices():
    """Get current available voices"""
    return voice_manager.get_voice_display_names()

def generate_audio_from_text(text, voice_display_name, use_chunking, chunk_size):
    """Generate audio from cleaned text"""
    if not text.strip():
        return None, [], "No text to convert to speech!"

    if not voice_display_name:
        return None, [], "Please select a voice!"

    try:
        voice_id = voice_manager.get_voice_id_from_display(voice_display_name)

        if use_chunking:
            # Split text into chunks and generate multiple audio files
            chunks = text_cleaner.split_into_chunks(text, chunk_size)
            print(f"[INFO] Splitting text into {len(chunks)} chunks")

            audio_files = tts.generate_audio_from_chunks(chunks, voice_id)

            if audio_files:
                return audio_files[0], audio_files, f"✅ Generated {len(audio_files)} audio files from {len(chunks)} text chunks with {voice_display_name}!"
            else:
                return None, [], "❌ Failed to generate any audio files!"
        else:
            # Generate single audio file
            if len(text) > 2000:  # Warn for long texts
                return None, [], "⚠️ Text is quite long for single file generation. Consider using chunking option!"

            audio_path = tts.generate_audio(text, voice_id)
            return audio_path, [audio_path], f"✅ Audio generated successfully with {voice_display_name}!"

    except Exception as e:
        return None, [], f"❌ Error generating audio: {str(e)}"

def upload_voice_sample(audio_file, voice_name):
    """Upload voice sample for cloning"""
    if not audio_file or not voice_name.strip():
        return "Please provide both audio file and voice name!", gr.Dropdown()

    try:
        # Create voice directory
        voice_id = re.sub(r'[^\w\s-]', '', voice_name.lower().replace(' ', '_'))
        voice_dir = os.path.join(VOICE_SAMPLES_DIR, voice_id)
        os.makedirs(voice_dir, exist_ok=True)

        # Save reference audio
        reference_path = os.path.join(voice_dir, "reference.wav")

        # Copy uploaded file to reference location
        import shutil
        shutil.copy2(audio_file, reference_path)

        # Add to voice manager
        voice_manager.add_voice(voice_id, voice_name, "Custom uploaded voice")

        # Update dropdown choices
        new_choices = get_available_voices()
        updated_dropdown = gr.Dropdown(choices=new_choices, value=new_choices[0] if new_choices else None)

        return f"✅ Voice '{voice_name}' uploaded successfully!", updated_dropdown

    except Exception as e:
        return f"❌ Error uploading voice: {str(e)}", gr.Dropdown()

def refresh_voices():
    """Refresh voice dropdown"""
    choices = get_available_voices()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

# --- Gradio App ---
with gr.Blocks(title="PDF to TTS Converter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 PDF to TTS Converter")
    gr.Markdown("Extract text from PDFs and convert to high-quality speech using AI voices")

    with gr.Tab("📄 PDF Processing"):
        gr.Markdown("## Upload and Process PDF")

        with gr.Row():
            pdf_input = gr.File(
                label="Upload PDF File",
                file_types=[".pdf"],
                type="filepath"
            )

            with gr.Column():
                extraction_method = gr.Dropdown(
                    choices=["auto", "pdfplumber", "pymupdf", "pypdf2"],
                    value="auto",
                    label="Extraction Method",
                    info="Auto tries methods in order of quality"
                )

                preserve_structure = gr.Checkbox(
                    label="Preserve Paragraph Structure",
                    value=True,
                    info="Keep paragraph breaks for better readability"
                )

        process_btn = gr.Button("📝 Extract & Clean Text", variant="primary", size="lg")

        processing_status = gr.Textbox(
            label="Processing Status",
            lines=3,
            placeholder="Upload a PDF and click 'Extract & Clean Text' to begin..."
        )

        extracted_text = gr.Textbox(
            label="Extracted & Cleaned Text",
            lines=15,
            placeholder="Processed text will appear here...",
            info="You can edit this text before generating audio"
        )

    with gr.Tab("🎙️ Audio Generation"):
        gr.Markdown("## Convert Text to Speech")

        with gr.Row():
            voice_dropdown = gr.Dropdown(
                label="Select Voice",
                choices=get_available_voices(),
                value=get_available_voices()[0] if get_available_voices() else None,
                info="Choose from standard or cloned voices"
            )

            refresh_btn = gr.Button("🔄 Refresh Voices", size="sm")

        with gr.Row():
            use_chunking = gr.Checkbox(
                label="Split into Chunks",
                value=True,
                info="Recommended for long texts to avoid memory issues"
            )

            chunk_size = gr.Slider(
                minimum=500,
                maximum=2000,
                value=1000,
                step=100,
                label="Chunk Size (characters)",
                info="Smaller chunks = more files, better quality"
            )

        audio_btn = gr.Button("🔊 Generate Audio", variant="secondary", size="lg")

        with gr.Row():
            first_audio = gr.Audio(
                label="First Audio File (Preview)",
                type="filepath"
            )

        all_audio_files = gr.File(
            label="All Generated Audio Files",
            file_count="multiple",
            type="filepath"
        )

        audio_status = gr.Textbox(
            label="Generation Status",
            lines=2,
            placeholder="Click 'Generate Audio' to convert text to speech..."
        )

    with gr.Tab("🎭 Voice Management"):
        gr.Markdown("## Upload Custom Voice Sample")
        gr.Markdown("Upload a clear audio sample (10-30 seconds) for voice cloning. WAV format recommended.")

        with gr.Row():
            upload_audio = gr.File(
                label="Audio Sample",
                file_types=[".wav", ".mp3", ".flac"],
                type="filepath"
            )

            upload_name = gr.Textbox(
                label="Voice Name",
                placeholder="Enter a name for this voice"
            )

        upload_btn = gr.Button("⬆️ Upload Voice", variant="primary")
        upload_status = gr.Textbox(label="Upload Status", lines=2)

        gr.Markdown("### 💡 Tips for Best Results:")
        gr.Markdown("""
        - **Audio Quality**: Use clear, high-quality recordings with minimal background noise
        - **Speaker**: Single speaker only, natural speaking pace
        - **Length**: 10-30 seconds of clear speech
        - **Format**: WAV files work best, but MP3 and FLAC are supported
        - **Content**: Natural conversational speech works better than reading
        """)

    with gr.Tab("ℹ️ Help & Settings"):
        gr.Markdown("## How to Use")
        gr.Markdown("""
        1. **Upload PDF**: Choose your PDF file in the 'PDF Processing' tab
        2. **Extract Text**: Click 'Extract & Clean Text' to process the document
        3. **Review Text**: Check and edit the extracted text if needed
        4. **Select Voice**: Choose a voice in the 'Audio Generation' tab
        5. **Generate Audio**: Click 'Generate Audio' to create speech files
        6. **Download**: Save the generated audio files to your device
        """)

        gr.Markdown("## Extraction Methods")
        gr.Markdown("""
        - **Auto**: Tries all methods automatically (recommended)
        - **PDFPlumber**: Best for complex layouts and tables
        - **PyMuPDF**: Good balance of speed and accuracy
        - **PyPDF2**: Fallback method for difficult files
        """)

        gr.Markdown("## Audio Chunking")
        gr.Markdown("""
        - **Enabled**: Splits long text into smaller pieces (recommended for books/papers)
        - **Disabled**: Generates one large audio file (only for short texts)
        - **Chunk Size**: Adjust based on your needs (1000 characters ≈ 2-3 minutes of audio)
        """)

    # Event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, extraction_method, preserve_structure, chunk_size],
        outputs=[extracted_text, processing_status]
    )

    audio_btn.click(
        fn=generate_audio_from_text,
        inputs=[extracted_text, voice_dropdown, use_chunking, chunk_size],
        outputs=[first_audio, all_audio_files, audio_status]
    )

    refresh_btn.click(
        fn=refresh_voices,
        outputs=voice_dropdown
    )

    upload_btn.click(
        fn=upload_voice_sample,
        inputs=[upload_audio, upload_name],
        outputs=[upload_status, voice_dropdown]
    )

if __name__ == "__main__":
    print("🚀 Starting PDF to TTS Converter...")
    print(f"📁 Voice samples directory: {VOICE_SAMPLES_DIR}")
    print(f"🔊 Generated audio directory: {GENERATED_AUDIO_DIR}")
    print(f"📄 Processed PDFs directory: {PROCESSED_PDFS_DIR}")

    # Check for required dependencies
    required_packages = ['PyPDF2', 'pymupdf', 'pdfplumber', 'speechbrain', 'TTS', 'num2words']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'pymupdf':
                missing_packages.append('PyMuPDF')
            else:
                missing_packages.append(package)

    if missing_packages:
        print("⚠️  Missing required packages. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nSpecific installation commands:")
        print("pip install PyPDF2 PyMuPDF pdfplumber speechbrain TTS num2words")
        print("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu")

    demo.launch(server_name="0.0.0.0", server_port=1602, share=False)