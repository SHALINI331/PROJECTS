import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk, ImageEnhance
import pytesseract
import pyttsx3
from fpdf import FPDF
import cv2
import numpy as np
import os
from datetime import datetime
import sys
from langdetect import detect, LangDetectException
import threading
from queue import Queue
import time
import hashlib
import re
from pathlib import Path
import shutil

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows

# Create secure temp directory for file operations
TEMP_DIR = os.path.join(os.path.expanduser("~"), ".visionvoice", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Configure maximum file sizes and batch limits for security
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB per file
MAX_BATCH_SIZE = 100 * 1024 * 1024  # 100MB total
MAX_BATCH_FILES = 20  # Maximum number of files in one batch

# Tesseract Configuration
try:
    # Ensure this path is correct for your system
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if not os.path.exists(tesseract_path):
        raise FileNotFoundError(f"Tesseract executable not found at {tesseract_path}")
    
    # Set Tesseract path
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Test if Tesseract is working and verify available languages
    try:
        pytesseract_version = pytesseract.get_tesseract_version()
        available_languages = pytesseract.get_languages(config='')
        print(f"Tesseract version: {pytesseract_version}")
        print(f"Available languages: {available_languages}")
        
        # Test OCR with a simple image
        test_img = np.zeros((100, 300), dtype=np.uint8)
        test_img = cv2.putText(test_img, 'Test', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        test_result = pytesseract.image_to_string(test_img).strip()
        print(f"OCR Test result: {test_result}")
        
        if not test_result:
            print("Warning: OCR test did not return any text. This might indicate a configuration issue.")
    except Exception as test_error:
        print(f"Tesseract test error: {str(test_error)}")
        raise
        
except Exception as e:
    error_msg = (
        f"Error initializing Tesseract: {str(e)}\n\n"
        "Please ensure:\n"
        "1. Tesseract is installed correctly (https://github.com/UB-Mannheim/tesseract/wiki)\n"
        "2. The installation path is correct\n"
        "3. Required language data files are installed\n"
        f"Checked path: {tesseract_path}"
    )
    print(error_msg)
    messagebox.showerror("Tesseract Error", error_msg)
    sys.exit(1)

class VisionVoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionVoice - OCR + Text-to-Speech")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)  # Set minimum window size
        
        # Set a base theme that works well for styling
        self.style = ttk.Style()
        try:
            # Try using 'clam' theme which is good for customization
            self.style.theme_use('clam')
        except tk.TclError:
            print("Clam theme not available, using default.")
            # Fallback to default if clam isn't available
            pass 
            
        # Nord theme colors
        self.nord_dark_bg = "#2E3440"
        self.nord_medium_bg = "#3B4252"
        self.nord_light_bg = "#4C566A"
        self.nord_text = "#ECEFF4"
        self.nord_accent_blue = "#88C0D0"
        self.nord_accent_frost = "#8FBCBB"
        self.nord_border = "#4C566A"
        self.nord_highlight = "#5E81AC"
        
        self.root.configure(bg=self.nord_dark_bg)
        
        # Initialize variables
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.text = ""
        self.engine = pyttsx3.init()
        self.speech_enabled = tk.BooleanVar(value=False)
        self.image = None
        self.original_image = None
        self.current_file_index = 0
        self.image_files = []
        self.extracted_texts = {}  # Store extracted text for each file
        self.detected_languages = {}  # Store detected language for each file
        self.processed_images = {}  # Store processed images for each file
        self.current_language = tk.StringVar(value='eng')  # Current language for speech
        self.processing_lock = threading.Lock()  # Lock for thread-safe operations
        self.current_file_var = tk.StringVar(value="No files loaded")  # Current file display
        
        # Expanded language mappings for better language support
        self.lang_codes = {
            'en': 'eng',     # English
            'ta': 'tam',     # Tamil
            'hi': 'hin',     # Hindi
            'te': 'tel',     # Telugu
            'kn': 'kan',     # Kannada
            'ml': 'mal',     # Malayalam
            'bn': 'ben',     # Bengali
            'gu': 'guj',     # Gujarati
            'pa': 'pan',     # Punjabi
            'or': 'ori',     # Odia
            'mr': 'mar',     # Marathi
            'ne': 'nep',     # Nepali
            'sa': 'san',     # Sanskrit
            'ur': 'urd',     # Urdu
            'ar': 'ara',     # Arabic
            'fa': 'fas',     # Persian
            'ja': 'jpn',     # Japanese
            'ko': 'kor',     # Korean
            'zh-cn': 'chi_sim', # Chinese Simplified
            'zh-tw': 'chi_tra', # Chinese Traditional
            'th': 'tha',     # Thai
            'ru': 'rus',     # Russian
            'fr': 'fra',     # French
            'de': 'deu',     # German
            'es': 'spa',     # Spanish
            'it': 'ita',     # Italian
            'pt': 'por',     # Portuguese
        }
        
        # Reverse mapping for display purposes
        self.lang_names = {v: k for k, v in self.lang_codes.items()}
        
        # Add friendly language display names
        self.language_display_names = {
            'eng': 'English',
            'tam': 'Tamil',
            'hin': 'Hindi',
            'tel': 'Telugu',
            'kan': 'Kannada',
            'mal': 'Malayalam',
            'ben': 'Bengali',
            'guj': 'Gujarati',
            'pan': 'Punjabi',
            'ori': 'Odia',
            'mar': 'Marathi',
            'nep': 'Nepali',
            'san': 'Sanskrit',
            'urd': 'Urdu',
            'ara': 'Arabic',
            'fas': 'Persian',
            'jpn': 'Japanese',
            'kor': 'Korean',
            'chi_sim': 'Chinese (Simplified)',
            'chi_tra': 'Chinese (Traditional)',
            'tha': 'Thai',
            'rus': 'Russian',
            'fra': 'French',
            'deu': 'German',
            'spa': 'Spanish',
            'ita': 'Italian',
            'por': 'Portuguese',
        }
        
        # Processing queue for background tasks
        self.process_queue = Queue()
        self.processing_thread = None
        
        # Setup application styles
        self.setup_styles()
        
        # Create UI components
        self.create_widgets()
        self.setup_menu()
        
        # Start processing thread
        self.start_processing_thread()
        
        # Setup autosave and cleanup
        self.setup_autosave()

    def setup_autosave(self):
        """Set up autosave and cleanup functionality"""
        # Clean old temporary files
        self.cleanup_temp_files()
        
        # Schedule periodic cleanup
        self.root.after(3600000, self.cleanup_temp_files)  # Cleanup every hour

    def cleanup_temp_files(self):
        """Clean up old temporary files (older than 24 hours)"""
        try:
            now = time.time()
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < now - 86400:
                    os.remove(file_path)
            print("Temporary file cleanup completed")
        except Exception as e:
            print(f"Error during temp file cleanup: {e}")
    
    def start_processing_thread(self):
        """Start the background processing thread"""
        self.processing_thread = threading.Thread(target=self.process_queue_items, daemon=True)
        self.processing_thread.start()

    def process_queue_items(self):
        """Process items in the queue"""
        while True:
            try:
                item = self.process_queue.get()
                if item is None:
                    break
                
                file_path = item
                self.process_single_file(file_path)
                self.process_queue.task_done()
                
                # Update progress after each file
                with self.processing_lock:
                    progress = (list(self.extracted_texts.keys()).index(file_path) + 1) / len(self.image_files)
                    self.progress_var.set(progress)
                    
                # Update UI status
                self.root.after(0, self.update_status, f"Processed {len(self.extracted_texts)}/{len(self.image_files)} files")
                
                # If all files processed, update UI with results
                if len(self.extracted_texts) == len(self.image_files):
                    self.root.after(0, self.update_ui_with_results)
                
            except Exception as e:
                self.root.after(0, self.update_status, f"Error processing file: {str(e)}")
                print(f"Processing error: {e}")

    def update_status(self, message):
        """Update status bar with message"""
        self.status_var.set(message)

    def validate_file(self, file_path):
        """Validate file before processing for security"""
        try:
            # Check file existence
            if not os.path.exists(file_path):
                return False, "File not found"
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > MAX_FILE_SIZE:
                return False, f"File too large ({file_size/1024/1024:.1f}MB > {MAX_FILE_SIZE/1024/1024}MB)"
                
            # Check file extension
            valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                return False, "Invalid file type"
                
            # Try opening as image
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                return False, "Invalid image file"
                
            return True, "File validated"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def process_single_file(self, file_path):
        """Process a single image file"""
        try:
            # Validate file
            valid, message = self.validate_file(file_path)
            if not valid:
                raise Exception(message)
            
            self.root.after(0, self.update_status, f"Processing: {os.path.basename(file_path)}...")
            
            # Load image
            image = Image.open(file_path)
            
            # Process image
            processed_text = self.extract_text(image)
            
            if not processed_text:
                raise Exception("No text could be extracted from the image")
            
            # Store results securely
            self.extracted_texts[file_path] = processed_text
            self.detected_languages[file_path] = 'eng'
            
            # Store processed image in temp directory for faster access
            temp_file = os.path.join(TEMP_DIR, f"{hashlib.md5(file_path.encode()).hexdigest()}.png")
            image.save(temp_file)
            
        except Exception as e:
            print(f"Processing error for {file_path}: {e}")
            # Still add to processed files to maintain order
            self.extracted_texts[file_path] = f"Error: {str(e)}"
            self.detected_languages[file_path] = "eng"  # Default to English on error

    def extract_text(self, image_input=None):
        """Optimized text extraction for score card text with numbers"""
        try:
            self.status_var.set("Extracting text...")
            self.progress_var.set(10)
            
            if image_input is None:
                if not self.image_files or self.current_file_index >= len(self.image_files):
                    messagebox.showinfo("No Image", "Please upload an image first")
                    return None
                current_file = self.image_files[self.current_file_index]
                img = Image.open(current_file)
            else:
                img = image_input
            
            try:
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img)
                
                # Process image
                enhanced = self.enhance_image(img_array)
                
                # OCR configurations optimized for score card
                configs = [
                    ('Score mode', '--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/():.,- "'),
                    ('Layout mode', '--oem 3 --psm 3 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/():.,- "'),
                    ('Single line', '--oem 3 --psm 7 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/():.,- "')
                ]
                
                best_text = ""
                best_conf = 0
                
                for config_name, config in configs:
                    try:
                        result = pytesseract.image_to_data(
                            enhanced,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Get text with confidence and position info
                        lines = []
                        current_line = []
                        last_top = -1
                        
                        for i, (text, conf, left, top) in enumerate(zip(result['text'], result['conf'], result['left'], result['top'])):
                            if not str(text).strip():
                                continue
                                
                            # New line detection
                            if top - last_top > 10 and last_top != -1:
                                if current_line:
                                    lines.append(' '.join(current_line))
                                current_line = []
                            
                            current_line.append(text)
                            last_top = top
                        
                        if current_line:
                            lines.append(' '.join(current_line))
                        
                        text = '\n'.join(lines)
                        
                        # Calculate confidence
                        confidences = [conf for conf in result['conf'] if conf > 0]
                        if confidences:
                            avg_conf = sum(confidences) / len(confidences)
                            if avg_conf > best_conf and text.strip():
                                best_conf = avg_conf
                                best_text = text
                    
                    except Exception as ocr_error:
                        print(f"Error with {config_name}: {str(ocr_error)}")
                
                if not best_text:
                    result = pytesseract.image_to_string(
                        enhanced,
                        config='--oem 3 --psm 3 -c preserve_interword_spaces=1'
                    )
                    if result.strip():
                        best_text = result
                    else:
                        raise Exception("No text could be extracted with any configuration")
                
                # Post-process score card specific text
                best_text = re.sub(r'\s+', ' ', best_text)  # Normalize spaces
                best_text = re.sub(r'(?<=\d)\s+(?=\/)', '', best_text)  # Fix split fractions
                best_text = re.sub(r'(?<=\d)\s+(?=%)', '', best_text)  # Fix split percentages
                
                # Clean and update UI
                cleaned_text = self.clean_extracted_text(best_text)
                self.root.after(0, self.update_text_display, cleaned_text)
                self.root.after(0, self.status_var.set, f"Text extraction complete (confidence: {best_conf:.1f}%)")
                self.root.after(0, self.progress_var.set, 100)
                
                if image_input is None and self.current_file_index < len(self.image_files):
                    current_file = self.image_files[self.current_file_index]
                    self.extracted_texts[current_file] = cleaned_text
                    self.detected_languages[current_file] = 'eng'
                
                return cleaned_text
                
            except Exception as img_error:
                raise Exception(f"Image processing error: {str(img_error)}")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Extraction error: {error_msg}")
            self.root.after(0, self.status_var.set, error_msg)
            self.root.after(0, self.progress_var.set, 0)
            return None

    def update_text_display(self, text):
        """Update text display with extracted text"""
        try:
            if not hasattr(self, 'text_box') or not self.text_box.winfo_exists():
                print("Warning: Text widget not available")
                return
                
            self.text_box.delete(1.0, tk.END)
            self.text_box.insert(tk.END, text)
            self.text_box.see(tk.END)
            self.root.update_idletasks()
        except Exception as e:
            print(f"Display update error: {str(e)}")

    def clean_extracted_text(self, text):
        """Clean and correct spelling in extracted text"""
        try:
            if not text:
                return ""

            # Expanded OCR error corrections
            replacements = {
                # Score card specific corrections
                'serby': 'set by',
                'rnezning': 'meaning',
                'vagabuiary': 'vocabulary',
                'ntening': 'listening',
                'unicaistond': 'understand',
                'ruige': 'range',
                'expresstony': 'expressions',
                'oonoqufoiame': 'monologue',
                'spekan': 'spoken',
                'confidene': 'confidence',
                'ffcnastes': 'indicates',
                'levei': 'level',
                'rintermecliate': 'intermediate',
                'guidatines': 'guidelines',
                'fansiyon': 'transition',
                'tintergrotnd': 'understand',
                'themoin': 'the main',
                'wmter': 'written',
                'infermediate': 'intermediate',
                'expiaimed': 'explained',
                'aeoorciing': 'according',
                'Shara': 'Share',
                'bezf': 'be',
                'B2': 'B2',  # Preserve B2 level notation
                '85%': '85%',  # Preserve percentage
                '90%': '90%',
                
                'mtermediate': 'intermediate',
                'frontian': 'transition',
                'cefr': 'CEFR',
                'abstact': 'abstract',
                'unfamniar': 'unfamiliar',
                'vocabuiary': 'vocabulary',
                'iistening': 'listening',
                'foiiow': 'follow',
                'confldence': 'confidence',
                'ievel': 'level',
                'expiained': 'explained',
                'ievei': 'level',
                'Bintermediate': 'B intermediate',
                'Bz': 'B2',
                'Gammon': 'Common',
                'nrontian': 'transition',
                'Fansiyon': 'transition',
                'Referenoe': 'Reference',
                'tintergrotond': 'understand',
                'themam': 'the main',
                'pomts': 'points',
                'wnter': 'written',
                'unfarnniar': 'unfamiliar',
                'vocabuiary': 'vocabulary',
                'Scora': 'Score',
                'Dce': 'Does',
                'expiamed': 'explained',
                'ffcnastes': 'indicates',
                'Rintermeciate': 'Intermediate',
                'Bz': 'B2',
                'Uppor': 'Upper',
                'aeoorcing': 'according',
                'guicatines': 'guidelines',
                'Gammon': 'Common',
                'Frontian': 'transition',
                'Framevork': 'Framework',
                'Referenoe': 'Reference',
                'tintergrotond': 'understand',
                'themam': 'the main',
                'pomts': 'points',
                'abstact': 'abstract',
                'wnter': 'written',
                'unfarnniar': 'unfamiliar',
                'vocabuiary': 'vocabulary',
                'Scora': 'Score',
                'Unicaistond': 'Understand',
                'ruige': 'range',
                'idiomatic': 'idiomatic',
                'expressiony': 'expressions',
                'foiiow': 'follow',
                'spekan': 'spoken',
                'confldence': 'confidence',
                
                # Common OCR fixes for numbers and special characters
                'B2': 'B2',  # Preserve B2 level notation
                '85%': '85%',  # Preserve percentage
                '90%': '90%',
                
                # Common word fixes
                'tho': 'the',
                'thai': 'that',
                'ls': 'is',
                'lt': 'it',
                'ln': 'in',
                'lf': 'if',
                'ofthe': 'of the',
                'andthe': 'and the',
                'tothe': 'to the',
                'forthe': 'for the',
                'withthe': 'with the',
                'fromthe': 'from the',
                'inthe': 'in the',
                'onthe': 'on the',
                'atthe': 'at the',
                
                # Letter confusions
                'l': 'i',      # lowercase L to i
                '0': 'o',      # zero to o
                '1': 'l',      # one to l
                'rn': 'm',     # 'rn' to 'm'
                'cl': 'd',     # 'cl' to 'd'
                'vv': 'w',     # 'vv' to 'w'
                'ii': 'n',     # 'ii' to 'n'
                
                # Number and symbol formatting
                r'\s+%': '%',      # Remove space before percentage
                r'\s+\.': '.',     # Remove space before period
                r'\s+,': ',',      # Remove space before comma
                r'\s+:': ':',      # Remove space before colon
                r'\s+;': ';',      # Remove space before semicolon
                r'\s+!': '!',      # Remove space before exclamation
                r'\s+\?': '?',     # Remove space before question mark
                r'\(\s+': '(',     # Remove space after opening parenthesis
                r'\s+\)': ')',     # Remove space before closing parenthesis
            }
            
            # Fix common OCR errors
            words = text.split()
            corrected_words = []
            
            for word in words:
                # Skip numbers and special characters
                if word.isdigit() or all(not c.isalnum() for c in word):
                    corrected_words.append(word)
                    continue
                
                # Apply common replacements
                corrected_word = word
                
                # Fix repeated character errors (e.g., 'hellooo' -> 'hello')
                corrected_word = re.sub(r'(.)\1{2,}', r'\1\1', corrected_word)
                
                # Apply word-level replacements
                for old, new in replacements.items():
                    if not old.startswith(r'\s'):  # Skip regex patterns
                        # Case-insensitive replacement
                        pattern = re.compile(re.escape(old), re.IGNORECASE)
                        corrected_word = pattern.sub(new, corrected_word)
                
                corrected_words.append(corrected_word)
            
            # Join words and fix spacing
            text = ' '.join(corrected_words)
            
            # Fix multiple spaces and line breaks
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # Fix common punctuation errors
            text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', '\n', text)  # Add line breaks after sentences
            text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)  # Fix broken words across lines
            text = re.sub(r'(\d)\s*\n\s*(\d)', r'\1\2', text)  # Fix broken numbers across lines
            
            # Fix spacing around punctuation
            for pattern in [r'\s+%', r'\s+\.', r'\s+,', r'\s+:', r'\s+;', r'\s+!', r'\s+\?']:
                text = re.sub(pattern, pattern[-1], text)
            
            # Fix common sentence structure issues
            text = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)  # Capitalize after sentence
            text = re.sub(r'\s+([.!?])', r'\1', text)  # Fix spaces before punctuation
            
            return text.strip()
            
        except Exception as e:
            print(f"Error in clean_extracted_text: {str(e)}")
            return text

    def update_ui_with_results(self):
        """Update UI with all processed results"""
        # Update text box with all processed texts
        self.text_box.delete('1.0', tk.END)
        
        # Sort files by their original order
        sorted_files = sorted(self.extracted_texts.keys(), 
                              key=lambda f: self.image_files.index(f) if f in self.image_files else 999)
        
        for path in sorted_files:
            text = self.extracted_texts[path]
            lang_code = self.detected_languages[path]
            lang_name = self.language_display_names.get(lang_code, lang_code)
            file_name = os.path.basename(path)
            
            # Add file header with styling
            self.text_box.insert(tk.END, f"File: ", "file_header")
            self.text_box.insert(tk.END, f"{file_name}\n", "file_name")
            self.text_box.insert(tk.END, f"Detected Language: ", "lang_header")
            self.text_box.insert(tk.END, f"{lang_name}\n\n", "lang_name")
            
            # Add the extracted text
            self.text_box.insert(tk.END, f"{text}\n", "content")
            
            # Add separator
            self.text_box.insert(tk.END, "=" * 50 + "\n\n", "separator")
            
        # Update current language for speech
        if self.image_files and self.current_file_index < len(self.image_files):
            current_file = self.image_files[self.current_file_index]
            if current_file in self.detected_languages:
                self.current_language.set(self.detected_languages[current_file])
                
        # Update progress bar to complete
        self.progress_var.set(1.0)
        self.status_var.set(f"Processed {len(self.extracted_texts)} files")
        
        # Display current file name
        if self.image_files and self.current_file_index < len(self.image_files):
            current_file = os.path.basename(self.image_files[self.current_file_index])
            self.current_file_var.set(f"File {self.current_file_index+1}/{len(self.image_files)}: {current_file}")
            
        # Display the current image
        self.show_current_image()

    def setup_styles(self):
        """Set up application styles with Nord theme"""
        # Configure ttk styles
        self.style.configure('.', 
                          background=self.nord_dark_bg, 
                          foreground=self.nord_text, 
                          font=('Segoe UI', 11))
        
        self.style.configure('TFrame', background=self.nord_dark_bg)
        self.style.configure('TLabel', 
                           background=self.nord_dark_bg, 
                           foreground=self.nord_text, 
                           font=('Segoe UI', 11))
        
        # Enhanced button styling
        self.style.configure('TButton', 
                          background=self.nord_highlight,
                          foreground=self.nord_text, 
                          font=('Segoe UI', 11, 'bold'), 
                          padding=10,
                          borderwidth=2,
                          relief='raised')
        self.style.map('TButton', 
                    background=[('active', self.nord_accent_frost), 
                              ('pressed', self.nord_accent_blue)],
                    foreground=[('active', self.nord_dark_bg)])

        # Title styling
        self.style.configure('Title.TLabel',
                          font=('Segoe UI', 28, 'bold'),
                          padding=15,
                          anchor='center')

        # Control frame styling
        self.style.configure('Controls.TFrame',
                          background=self.nord_medium_bg,
                          padding=15,
                          relief='groove',
                          borderwidth=2)

        # Main panel styling
        self.style.configure('MainPanel.TFrame',
                          background=self.nord_medium_bg,
                          relief='solid',
                          borderwidth=2)
                          
        # Checkbox styling
        self.style.configure('TCheckbutton',
                          background=self.nord_medium_bg, 
                          foreground=self.nord_text,
                          font=('Segoe UI', 11),
                          padding=5)
        self.style.map('TCheckbutton',
                      background=[('active', self.nord_light_bg)],
                      foreground=[('active', self.nord_text)])
                      
        # Combobox styling
        self.style.map("TCombobox",
                    fieldbackground=[('readonly', self.nord_light_bg)],
                    selectbackground=[('readonly', self.nord_highlight)],
                    selectforeground=[('readonly', self.nord_text)],
                    background=[('readonly', self.nord_light_bg)])
                    
        # Progressbar styling
        self.style.configure("TProgressbar", 
                          background=self.nord_accent_blue,
                          troughcolor=self.nord_medium_bg,
                          borderwidth=0,
                          thickness=10)
        
        # Configure Menu with enhanced styling
        self.root.option_add("*Menu.background", self.nord_medium_bg)
        self.root.option_add("*Menu.foreground", self.nord_text)
        self.root.option_add("*Menu.activeBackground", self.nord_highlight)
        self.root.option_add("*Menu.activeForeground", self.nord_text)
        self.root.option_add("*Menu.font", ('Segoe UI', 11))
        self.root.option_add("*Menu.relief", 'solid')
        self.root.option_add("*Menu.borderWidth", 1)

    def setup_menu(self):
        """Set up application menu with enhanced options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Single Image", command=self.upload_image)
        file_menu.add_command(label="Open Multiple Images", command=self.upload_images)
        file_menu.add_command(label="Capture Webcam", command=self.capture_webcam)
        file_menu.add_separator()
        file_menu.add_command(label="Save as PDF", command=self.save_as_pdf)
        file_menu.add_command(label="Save Text Only", command=self.save_text_only)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Text", command=self.clear_text)
        edit_menu.add_command(label="Copy Text", command=self.copy_text)
        edit_menu.add_command(label="Select All", command=self.select_all_text)
        
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Extract Text", command=self.extract_text)
        process_menu.add_command(label="Enhance Image", command=self.enhance_image_button)
        process_menu.add_command(label="Batch Process All", command=self.batch_process)
        
        language_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Language", menu=language_menu)
        
        # Sort languages by display name
        sorted_langs = sorted(self.language_display_names.items(), key=lambda x: x[1])
        
        for lang_code, lang_name in sorted_langs:
            # Check if this language is installed
            available_langs = pytesseract.get_languages(config='')
            if lang_code in available_langs:
                language_menu.add_command(
                    label=f"{lang_name} ({lang_code})",
                    command=lambda lc=lang_code: self.set_current_language(lc)
                )
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Language Support", command=self.show_language_info)

    def create_widgets(self):
        """Create application UI widgets with enhanced user experience"""
        # Title
        title_frame = tk.Frame(self.root, bg=self.nord_dark_bg)
        title_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        title_label = ttk.Label(title_frame,
                               text="VisionVoice OCR",
                               style='Title.TLabel',
                               anchor='center')
        title_label.pack(expand=True, fill=tk.X)

        # Main container
        main_container = ttk.Frame(self.root, padding=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)

        # --- Left Panel --- 
        left_panel = ttk.Frame(main_container, style='MainPanel.TFrame', padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.rowconfigure(0, weight=0) # Navigation area
        left_panel.rowconfigure(1, weight=1) # Image frame expands
        left_panel.rowconfigure(2, weight=0) # Controls frame doesn't expand
        left_panel.columnconfigure(0, weight=1)

        # Image navigation area
        nav_frame = ttk.Frame(left_panel, style='Controls.TFrame')
        nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        nav_frame.columnconfigure(0, weight=1)
        nav_frame.columnconfigure(1, weight=1)
        nav_frame.columnconfigure(2, weight=1)
        
        ttk.Button(nav_frame, text="⬅️ Previous", command=self.previous_image).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Label(nav_frame, textvariable=self.current_file_var, anchor='center').grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(nav_frame, text="Next ➡️", command=self.next_image).grid(row=0, column=2, padx=5, pady=5, sticky='e')

        # Image display area
        img_frame = ttk.Frame(left_panel, relief='sunken', borderwidth=1)
        img_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        img_frame.columnconfigure(0, weight=1)
        img_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(img_frame, bg=self.nord_medium_bg, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Image controls
        controls_frame = ttk.Frame(left_panel, style='Controls.TFrame')
        controls_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        
        ttk.Button(controls_frame, text="Upload Image", command=self.upload_image).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(controls_frame, text="Extract Text", command=self.extract_text).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(controls_frame, text="Enhance Image", command=self.enhance_image_button).pack(side=tk.LEFT, padx=5, pady=5)
        
        # --- Right Panel ---
        right_panel = ttk.Frame(main_container, style='MainPanel.TFrame', padding=10)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(0, weight=1)  # Text area expands
        right_panel.rowconfigure(1, weight=0)  # Controls don't expand
        right_panel.columnconfigure(0, weight=1)
        
        # Text display area with custom styling
        self.text_box = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, 
                                                font=('Segoe UI', 11),
                                                bg=self.nord_medium_bg,
                                                fg=self.nord_text,
                                                insertbackground=self.nord_text,
                                                selectbackground=self.nord_highlight,
                                                selectforeground=self.nord_text,
                                                relief="sunken",
                                                borderwidth=1,
                                                padx=10, pady=10)
        self.text_box.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Configure text styles
        self.text_box.tag_configure("file_header", font=('Segoe UI', 11, 'bold'), foreground="#88C0D0")
        self.text_box.tag_configure("file_name", font=('Segoe UI', 11), foreground="#ECEFF4")
        self.text_box.tag_configure("lang_header", font=('Segoe UI', 11, 'bold'), foreground="#88C0D0")
        self.text_box.tag_configure("lang_name", font=('Segoe UI', 11), foreground="#ECEFF4")
        self.text_box.tag_configure("content", font=('Segoe UI', 11), foreground="#ECEFF4")
        self.text_box.tag_configure("separator", font=('Segoe UI', 11), foreground="#4C566A")
        
        # Text controls
        text_controls = ttk.Frame(right_panel, style='Controls.TFrame')
        text_controls.grid(row=1, column=0, sticky="ew")
        text_controls.columnconfigure(0, weight=1)
        text_controls.columnconfigure(1, weight=1)
        text_controls.columnconfigure(2, weight=1)
        
        # Speech controls
        speech_frame = ttk.Frame(text_controls)
        speech_frame.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        ttk.Checkbutton(speech_frame, text="Enable Speech", 
                      variable=self.speech_enabled,
                      style='TCheckbutton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(speech_frame, text="Speak Text", 
                 command=self.speak_text).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(speech_frame, text="Stop Speech", 
                 command=self.stop_speech).pack(side=tk.LEFT, padx=5)
                 
        # Save controls
        save_frame = ttk.Frame(text_controls)
        save_frame.grid(row=0, column=1, sticky="e", padx=5, pady=5)
        
        ttk.Button(save_frame, text="Save as PDF", 
                 command=self.save_as_pdf).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(save_frame, text="Save Text", 
                 command=self.save_text_only).pack(side=tk.LEFT, padx=5)
                 
        # Clear control
        clear_frame = ttk.Frame(text_controls)
        clear_frame.grid(row=0, column=2, sticky="e", padx=5, pady=5)
        
        ttk.Button(clear_frame, text="Clear All", 
                 command=self.clear_text).pack(side=tk.RIGHT, padx=5)
        
        # Status bar with progress
        status_frame = ttk.Frame(self.root, relief="sunken", padding=(10, 5))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Progressbar(status_frame, variable=self.progress_var, length=200, mode='determinate').pack(side=tk.RIGHT, padx=5)

    def upload_image(self):
        """Upload a single image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), 
                     ("All files", "*.*"))
        )
        
        if file_path:
            # Clear previous data
            self.clear_data()
            
            # Add file to list
            self.image_files = [file_path]
            self.current_file_index = 0
            self.current_file_var.set(f"File: {os.path.basename(file_path)}")
            
            # Show the image
            self.show_current_image()
            
            # Process the file
            self.process_queue.put(file_path)
            self.status_var.set(f"Processing: {os.path.basename(file_path)}")
            self.progress_var.set(0.5)

    def upload_images(self):
        """Upload multiple image files"""
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), 
                     ("All files", "*.*"))
        )
        
        if file_paths:
            # Check batch size limits
            total_size = sum(os.path.getsize(f) for f in file_paths)
            if len(file_paths) > MAX_BATCH_FILES:
                messagebox.showwarning(
                    "Batch Size Exceeded", 
                    f"Maximum {MAX_BATCH_FILES} files can be processed at once. " +
                    f"Please select fewer files."
                )
                return
                
            if total_size > MAX_BATCH_SIZE:
                messagebox.showwarning(
                    "Batch Size Exceeded", 
                    f"Total size exceeds {MAX_BATCH_SIZE/1024/1024}MB. " +
                    f"Please select smaller files."
                )
                return
            
            # Clear previous data
            self.clear_data()
            
            # Add files to list
            self.image_files = list(file_paths)
            self.current_file_index = 0
            
            if self.image_files:
                self.current_file_var.set(f"File 1/{len(self.image_files)}: {os.path.basename(self.image_files[0])}")
                
                # Show the first image
                self.show_current_image()
                
                # Start batch processing
                self.batch_process()

    def capture_webcam(self):
        """Capture image from webcam"""
        try:
            # Initialize the webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open webcam")
                return
                
            # Create a new top-level window for webcam preview
            webcam_window = tk.Toplevel(self.root)
            webcam_window.title("Webcam Capture")
            webcam_window.geometry("800x600")
            webcam_window.configure(bg=self.nord_dark_bg)
            
            # Canvas for displaying webcam feed
            webcam_canvas = tk.Canvas(webcam_window, bg="black")
            webcam_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Control panel
            control_panel = ttk.Frame(webcam_window, style="Controls.TFrame")
            control_panel.pack(fill=tk.X, pady=10, padx=10)
            
            captured_image = None
            
            def update_webcam():
                nonlocal webcam_window
                ret, frame = cap.read()
                
                if ret:
                    # Convert to RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    # Resize while maintaining aspect ratio
                    width, height = webcam_canvas.winfo_width(), webcam_canvas.winfo_height()
                    img.thumbnail((width, height))
                    
                    # Convert to PhotoImage
                    img_tk = ImageTk.PhotoImage(image=img)
                    
                    # Update canvas
                    webcam_canvas.delete("all")
                    webcam_canvas.create_image(width//2, height//2, image=img_tk)
                    webcam_canvas.image = img_tk  # Keep a reference
                    
                    # Continue updating while window exists
                    if webcam_window.winfo_exists():
                        webcam_window.after(10, update_webcam)
                    else:
                        cap.release()
            
            def capture_image():
                nonlocal captured_image
                ret, frame = cap.read()
                
                if ret:
                    # Convert to RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    captured_image = Image.fromarray(frame_rgb)
                    
                    # Show captured notification
                    webcam_canvas.delete("notification")
                    webcam_canvas.create_text(
                        webcam_canvas.winfo_width()//2,
                        webcam_canvas.winfo_height()//2,
                        text="Image Captured!",
                        fill="white",
                        font=("Segoe UI", 24, "bold"),
                        tags="notification")
                    
                    # Enable use button
                    use_button.config(state=tk.NORMAL)
                    
                    # Flash notification
                    webcam_window.after(1000, lambda: webcam_canvas.delete("notification"))
            
            def use_captured_image():
                nonlocal captured_image
                
                if captured_image:
                    # Save to a temporary file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_path = os.path.join(TEMP_DIR, f"webcam_{timestamp}.png")
                    captured_image.save(temp_path)
                    
                    # Close webcam window
                    webcam_window.destroy()
                    cap.release()
                    
                    # Process the captured image
                    self.clear_data()
                    self.image_files = [temp_path]
                    self.current_file_index = 0
                    self.current_file_var.set(f"File: Webcam Capture {timestamp}")
                    self.show_current_image()
                    self.process_queue.put(temp_path)
            
            # Control buttons
            capture_button = ttk.Button(control_panel, text="Capture", command=capture_image)
            capture_button.pack(side=tk.LEFT, padx=10)
            
            use_button = ttk.Button(control_panel, text="Use Image", 
                                  command=use_captured_image, state=tk.DISABLED)
            use_button.pack(side=tk.LEFT, padx=10)
            
            cancel_button = ttk.Button(control_panel, text="Cancel", 
                                     command=lambda: (cap.release(), webcam_window.destroy()))
            cancel_button.pack(side=tk.RIGHT, padx=10)
            
            # Start updating webcam feed
            webcam_window.after(100, update_webcam)  # Give time for canvas to initialize
        
        except Exception as e:
            messagebox.showerror("Webcam Error", f"Error accessing webcam: {str(e)}")

    def clear_data(self):
        """Clear all data"""
        self.image_files = []
        self.current_file_index = 0
        self.extracted_texts = {}
        self.detected_languages = {}
        self.processed_images = {}
        self.text_box.delete('1.0', tk.END)
        self.image = None
        self.original_image = None
        self.canvas.delete("all")
        self.current_file_var.set("No files loaded")
        self.progress_var.set(0.0)

    def show_current_image(self):
        """Show current image in canvas"""
        try:
            if not self.image_files or self.current_file_index >= len(self.image_files):
                self.canvas.delete("all")
                return
                
            current_file = self.image_files[self.current_file_index]
            
            # Try to use processed image if available
            if current_file in self.processed_images:
                display_image = self.processed_images[current_file]
            else:
                # Load the original image
                display_image = Image.open(current_file)
                
            # Store original for processing
            self.original_image = display_image.copy()
            
            # Resize for display
            display_image = self.resize_image_for_display(display_image)
            
            # Create photoimage
            self.image = ImageTk.PhotoImage(display_image)
            
            # Show on canvas
            self.canvas.delete("all")
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Center the image
            x = max(0, (canvas_width - self.image.width()) // 2)
            y = max(0, (canvas_height - self.image.height()) // 2)
            
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.image)
            
            # Update current file display
            if self.image_files:
                self.current_file_var.set(
                    f"File {self.current_file_index+1}/{len(self.image_files)}: {os.path.basename(current_file)}"
                )
                
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2, 
                self.canvas.winfo_height()//2,
                text=f"Error displaying image:\n{str(e)}",
                fill=self.nord_text,
                font=('Segoe UI', 11)
            )

    def resize_image_for_display(self, img, max_width=800, max_height=600):
        """Resize image while maintaining aspect ratio"""
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width() or max_width
        canvas_height = self.canvas.winfo_height() or max_height
        
        # Use canvas dimensions if available, otherwise default
        max_width = min(max_width, canvas_width)
        max_height = min(max_height, canvas_height)
        
        # Get image dimensions
        width, height = img.size
        
        # Calculate scale factor
        scale_w = max_width / width if width > max_width else 1
        scale_h = max_height / height if height > max_height else 1
        scale = min(scale_w, scale_h)
        
        # Only resize if necessary
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return img.resize((new_width, new_height), Image.LANCZOS)
        
        return img

    def previous_image(self):
        """Show previous image in list"""
        if self.image_files and len(self.image_files) > 1:
            self.current_file_index = (self.current_file_index - 1) % len(self.image_files)
            self.show_current_image()
            
            # Update text display if available
            self.show_current_text()

    def next_image(self):
        """Show next image in list"""
        if self.image_files and len(self.image_files) > 1:
            self.current_file_index = (self.current_file_index + 1) % len(self.image_files)
            self.show_current_image()
            
            # Update text display if available
            self.show_current_text()

    def show_current_text(self):
        """Show text for current image"""
        if not self.image_files or self.current_file_index >= len(self.image_files):
            return
        
        current_file = self.image_files[self.current_file_index]
        
        if current_file in self.extracted_texts:
            # Update language for speech
            if current_file in self.detected_languages:
                self.current_language.set(self.detected_languages[current_file])
                
            # If we have only one file, update everything
            if len(self.image_files) == 1:
                self.update_ui_with_results()
                return
                
            # Otherwise, scroll to the current file section
            self.text_box.tag_remove("current", "1.0", tk.END)
            
            # Search for the current file
            text_content = self.text_box.get("1.0", tk.END)
            file_name = os.path.basename(current_file)
            file_index = text_content.find(f"File: {file_name}")
            
            if file_index >= 0:
                # Convert character position to line.column
                line_start = text_content[:file_index].count('\n') + 1
                self.text_box.see(f"{line_start}.0")
                
                # Highlight current section
                end_index = text_content.find("="*50, file_index)
                if end_index >= 0:
                    line_end = text_content[:end_index].count('\n') + 1
                    self.text_box.tag_add("current", f"{line_start}.0", f"{line_end}.0")
                    self.text_box.tag_configure("current", background=self.nord_highlight)
                    
                    # Ensure the highlighted section is visible
                    self.text_box.see(f"{line_start}.0")
                    
                    # Update any UI elements that should reflect the current file's text
                    self.update_status_bar(f"Showing text for: {file_name}")
                else:
                    # If we can't find the end marker, just highlight the line with the filename
                    self.text_box.tag_add("current", f"{line_start}.0", f"{line_start}.end")
                    self.text_box.tag_configure("current", background=self.nord_highlight)
                    self.update_status_bar(f"Partial view for: {file_name}")

    def update_status_bar(self, message):
        """Update status bar with message"""
        self.status_var.set(message)

    def batch_process(self):
        """Process all files in batch mode"""
        if not self.image_files:
            return
        
        self.status_var.set(f"Processing batch: 0/{len(self.image_files)}")
        self.progress_var.set(0)
        
        # Clear results
        self.text_box.delete('1.0', tk.END)
        
        # Process each file
        for i, file_path in enumerate(self.image_files):
            # Add to process queue
            self.process_queue.put(file_path)
            
            # Update status
            self.status_var.set(f"Processing batch: {i+1}/{len(self.image_files)}")
            self.progress_var.set((i+1) / len(self.image_files) * 100)
            
        # Final update will happen when processing is complete

    def speak_text(self):
        """Speak the extracted text in the appropriate language"""
        if not self.speech_enabled.get():
            messagebox.showinfo("Speech Disabled", "Please enable speech first")
            return
            
        current_file = self.image_files[self.current_file_index]
        if current_file not in self.extracted_texts:
            messagebox.showinfo("No Text", "Please extract text first")
            return
            
        text = self.extracted_texts[current_file]
        lang = self.detected_languages.get(current_file, 'eng')
        
        if not text.strip():
            messagebox.showinfo("Empty Text", "No text to speak")
            return
            
        try:
            engine = pyttsx3.init()
            
            # Configure properties based on language
            if lang == 'tam':
                # Configure for Tamil
                engine.setProperty('rate', 130)    # Slower rate for Tamil
                # Try to find Tamil voice if available
                voices = engine.getProperty('voices')
                tamil_voice = None
                for voice in voices:
                    if 'tamil' in voice.name.lower():
                        tamil_voice = voice.id
                        break
                if tamil_voice:
                    engine.setProperty('voice', tamil_voice)
            else:
                # Configure for English
                engine.setProperty('rate', 150)
                voices = engine.getProperty('voices')
                english_voice = None
                for voice in voices:
                    if 'english' in voice.name.lower():
                        english_voice = voice.id
                        break
                if english_voice:
                    engine.setProperty('voice', english_voice)
            
            engine.setProperty('volume', 0.9)
            
            self.status_var.set(f"Speaking text in {'Tamil' if lang == 'tam' else 'English'}...")
            engine.say(text)
            engine.runAndWait()
            self.status_var.set("Speech complete")
            
        except Exception as e:
            self.status_var.set(f"Speech error: {str(e)}")
            messagebox.showerror("Speech Error", str(e))
        finally:
            try:
                engine.stop()
            except:
                pass

    def stop_speech(self):
        """Stop current speech"""
        try:
            if hasattr(self, 'engine'):
                self.engine.stop()
            self.status_var.set("Speech stopped")
        except Exception as e:
            print(f"Error stopping speech: {e}")

    def save_as_pdf(self):
        """Save extracted text and image as PDF"""
        if not self.image_files or not any(f in self.extracted_texts for f in self.image_files):
            messagebox.showinfo("No Content", "Please extract text from at least one image first")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save as PDF"
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Creating PDF...")
            self.progress_var.set(10)
            
            # Create PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Add title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "OCR Text Extraction Results", 0, 1, 'C')
            pdf.ln(10)
            
            # Process each file
            for i, file_path in enumerate(self.image_files):
                if file_path not in self.extracted_texts:
                    continue
                
                # Update progress
                self.progress_var.set(10 + (i / len(self.image_files)) * 80)
                
                if i > 0:
                    pdf.add_page()
                
                # Add file info
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, f"File: {os.path.basename(file_path)}", 0, 1)
                
                # Add language info
                language = self.detected_languages.get(file_path, "Unknown")
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 10, f"Language: {language}", 0, 1)
                
                # Add image
                try:
                    # Use processed image if available, otherwise original
                    if file_path in self.processed_images:
                        img = self.processed_images[file_path]
                    else:
                        img = Image.open(file_path)
                    
                    # Save to temporary file
                    temp_img_path = os.path.join(TEMP_DIR, "temp_img.png")
                    img = img.convert('RGB')
                    img.save(temp_img_path)
                    
                    # Resize for PDF
                    img_width = min(pdf.w - 40, 160)  # Max width in PDF
                    pdf.image(temp_img_path, x=10, w=img_width)
                    
                    # Remove temp file
                    os.remove(temp_img_path)
                except Exception as e:
                    pdf.set_text_color(255, 0, 0)
                    pdf.cell(0, 10, f"Error adding image: {str(e)}", 0, 1)
                    pdf.set_text_color(0, 0, 0)
                
                pdf.ln(10)
                
                # Add text content
                text = self.extracted_texts[file_path]
                pdf.set_font("Arial", '', 10)
                
                # Split text into lines and add to PDF
                for line in text.split('\n'):
                    pdf.multi_cell(0, 5, line)
                
                pdf.ln(5)
            
            # Save PDF
            pdf.output(file_path)
            
            self.progress_var.set(100)
            self.status_var.set(f"PDF saved to {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving PDF: {str(e)}")
            messagebox.showerror("PDF Error", f"Error saving PDF: {str(e)}")
            self.progress_var.set(0)

    def save_text_only(self):
        """Save extracted text to a text file"""
        if not self.image_files or not any(f in self.extracted_texts for f in self.image_files):
            messagebox.showinfo("No Content", "Please extract text from at least one image first")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Text"
        )
        
        if not file_path:
            return
        
        try:
            self.status_var.set("Saving text...")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for i, img_path in enumerate(self.image_files):
                    if img_path not in self.extracted_texts:
                        continue
                    
                    if i > 0:
                        f.write("\n" + "="*50 + "\n\n")
                    
                    f.write(f"File: {os.path.basename(img_path)}\n\n")
                    
                    language = self.detected_languages.get(img_path, "Unknown")
                    f.write(f"Language: {language}\n\n")
                    
                    f.write(self.extracted_texts[img_path])
            
            self.status_var.set(f"Text saved to {file_path}")
            
        except Exception as e:
            self.status_var.set(f"Error saving text: {str(e)}")
            messagebox.showerror("Save Error", f"Error saving text: {str(e)}")

    def clear_text(self):
        """Clear the text box"""
        self.text_box.delete('1.0', tk.END)
        self.status_var.set("Text cleared")

    def copy_text(self):
        """Copy selected text to clipboard"""
        try:
            selected_text = self.text_box.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.status_var.set("Text copied to clipboard")
        except tk.TclError:  # No selection
            # If no text is selected, copy all text
            all_text = self.text_box.get('1.0', tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(all_text)
            self.status_var.set("All text copied to clipboard")

    def select_all_text(self):
        """Select all text in the text box"""
        self.text_box.tag_add(tk.SEL, "1.0", tk.END)
        self.text_box.mark_set(tk.INSERT, "1.0")
        self.text_box.see(tk.INSERT)
        self.status_var.set("All text selected")

    def show_about(self):
        """Show about dialog"""
        about_text = """VisionVoice OCR
Version 1.0

A powerful OCR and Text-to-Speech application that supports multiple languages.

Features:
- Image to text conversion
- Multi-language support
- Text-to-speech
- Image enhancement
- Batch processing
- PDF export

Created with Python using Tesseract OCR."""

        messagebox.showinfo("About VisionVoice", about_text)

    def show_language_info(self):
        """Show supported languages information"""
        try:
            available_langs = pytesseract.get_languages(config='')
            
            lang_info = "Supported Languages:\n\n"
            for lang_code in sorted(available_langs):
                lang_name = self.language_display_names.get(lang_code, lang_code)
                lang_info += f"• {lang_name} ({lang_code})\n"
            
            messagebox.showinfo("Language Support", lang_info)
        except Exception as e:
            messagebox.showerror("Error", f"Could not retrieve language information: {str(e)}")

    def set_current_language(self, lang_code):
        """Set the current language for OCR and speech"""
        self.current_language.set(lang_code)
        self.status_var.set(f"Language set to: {self.language_display_names.get(lang_code, lang_code)}")

    def enhance_image(self, img_array):
        """Enhanced image processing for score card text with numbers"""
        try:
            # 1. Initial resize if needed
            height, width = img_array.shape
            max_dimension = 2000
            if width > max_dimension or height > max_dimension:
                scale = min(max_dimension / width, max_dimension / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # 2. Normalize and enhance contrast
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
            
            # 3. Apply CLAHE with larger tile size for better text contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            img_array = clahe.apply(img_array)
            
            # 4. Light denoising to preserve text edges
            img_array = cv2.fastNlMeansDenoising(img_array, None, h=5)
            
            # 5. Sharpen to enhance text edges
            kernel = np.array([[-1,-1,-1], [-1,9.5,-1], [-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
            
            # 6. Otsu's thresholding for clean text
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return img_array
            
        except Exception as e:
            print(f"Image enhancement error: {str(e)}")
            return img_array

    def enhance_image_button(self):
        """Handle enhance image button click"""
        try:
            if not self.image_files or self.current_file_index >= len(self.image_files):
                messagebox.showinfo("No Image", "Please upload an image first")
                return
            
            current_file = self.image_files[self.current_file_index]
            
            # Load and process image
            img = Image.open(current_file)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img = img.convert('L')  # Convert to grayscale
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Enhance the image
            enhanced = self.enhance_image(img_array)
            
            # Save enhanced image
            enhanced_path = os.path.join(TEMP_DIR, "enhanced_preview.png")
            cv2.imwrite(enhanced_path, enhanced)
            
            # Update the image display
            self.display_image(enhanced_path)
            self.status_var.set("Image enhanced")
            
        except Exception as e:
            error_msg = f"Enhancement error: {str(e)}"
            print(error_msg)
            self.status_var.set(error_msg)

def main():
    # Create temp directory if it doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    # Create and run app
    root = tk.Tk()
    app = VisionVoiceApp(root)
    root.mainloop()

    print(" VisionVoice main function running.")
    print(f"Using temp directory: {TEMP_DIR}")


if __name__ == "__main__":
    main()