from .base_processor import BaseProcessor
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional
from PIL import Image
import re

# Try to import advanced OCR dependencies, fallback to basic implementation
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    ADVANCED_OCR_AVAILABLE = True
except ImportError:
    ADVANCED_OCR_AVAILABLE = False
    print("⚠️  Advanced OCR dependencies not available, using basic implementation")

# Try to import Tesseract as fallback OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class ShortTextProcessor(BaseProcessor):
    def __init__(self,
                 model_id='microsoft/Florence-2-large',
                 use_gpu=True,
                 max_text_length=200,
                 task_prompt='<OCR>'):
        """
        Initialize Short Text processor optimized for quick text extraction
        
        Args:
            model_id (str): Florence model identifier  
            use_gpu (bool): Whether to use GPU acceleration
            max_text_length (int): Maximum expected text length for optimization
            task_prompt (str): OCR task prompt for Florence model
        """
        super().__init__()
        self.task_prompt = task_prompt
        self.max_text_length = max_text_length
        self.ocr_method = "basic"  # Will be updated based on available dependencies
        
        # Try to initialize advanced OCR first
        if ADVANCED_OCR_AVAILABLE:
            self._init_florence_ocr(model_id, use_gpu)
        elif TESSERACT_AVAILABLE:
            self._init_tesseract_ocr()
        else:
            self._init_basic_ocr()
        
        print(f"✅ ShortTextProcessor initialized using {self.ocr_method} OCR method")

    def _init_florence_ocr(self, model_id, use_gpu):
        """Initialize Florence-2 model for advanced OCR"""
        try:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype='auto'
            ).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.ocr_method = "florence"
            print(f"✅ Florence-2 OCR initialized on {self.device}")
        except Exception as e:
            print(f"⚠️  Could not load Florence model {model_id}: {e}")
            if TESSERACT_AVAILABLE:
                self._init_tesseract_ocr()
            else:
                self._init_basic_ocr()

    def _init_tesseract_ocr(self):
        """Initialize Tesseract OCR as fallback"""
        try:
            # Test if tesseract is working
            pytesseract.get_tesseract_version()
            self.ocr_method = "tesseract"
            print("✅ Tesseract OCR initialized")
        except Exception as e:
            print(f"⚠️  Tesseract not available: {e}")
            self._init_basic_ocr()

    def _init_basic_ocr(self):
        """Initialize basic OCR using OpenCV-based text detection"""
        self.ocr_method = "basic"
        print("✅ Basic OCR initialized (pattern-based text detection)")

    def _preprocess_image_for_ocr(self, pil_image):
        """
        Preprocess image to improve OCR accuracy for short text
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Convert to numpy for preprocessing
        img_array = np.array(pil_image)
        
        # Convert to grayscale for better OCR
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply adaptive thresholding to improve text clarity
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise the image
        denoised = cv2.medianBlur(binary, 3)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)

    def _run_florence_ocr(self, pil_image):
        """Run Florence OCR optimized for short text extraction"""
        if self.ocr_method != "florence":
            return ""
            
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_image_for_ocr(pil_image)
            
            # Prepare inputs for Florence
            inputs = self.processor(
                text=self.task_prompt,
                images=processed_image,
                return_tensors="pt"
            ).to(self.device, torch.float16)

            # Generate results with optimized parameters for short text
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=min(512, self.max_text_length + 50),
                do_sample=False,
                num_beams=1,
                early_stopping=True
            )

            # Process output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Extract and clean the text result
            result = self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(pil_image.width, pil_image.height)
            )
            
            # Extract text from Florence result
            if result and self.task_prompt in result:
                extracted_text = result[self.task_prompt]
                return self._clean_short_text(extracted_text)
            
            return ""
            
        except Exception as e:
            print(f"Error in Florence OCR: {e}")
            return ""

    def _run_tesseract_ocr(self, pil_image):
        """Run Tesseract OCR for text extraction"""
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_image_for_ocr(pil_image)
            
            # Configure Tesseract for short text
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?()-'
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(processed_image, config=config)
            return self._clean_short_text(text)
            
        except Exception as e:
            print(f"Error in Tesseract OCR: {e}")
            return ""

    def _run_basic_ocr(self, cv_image):
        """Run basic pattern-based text detection"""
        try:
            # Convert to grayscale
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image
            
            # Apply text detection using basic computer vision
            # This is a simple approach for demo purposes
            text_regions = self._detect_text_regions(gray)
            
            if text_regions:
                return f"Detected {len(text_regions)} text region(s) [Basic OCR Mode]"
            else:
                return "Short text processor ready [Basic OCR Mode]"
                
        except Exception as e:
            print(f"Error in basic OCR: {e}")
            return "Short text processor error"

    def _detect_text_regions(self, gray_image):
        """Basic text region detection using OpenCV"""
        try:
            # Apply morphological operations to detect text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            grad = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
            
            # Apply threshold to get binary image
            _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Connect horizontally oriented regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio to find text-like regions
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    rect = cv2.boundingRect(contour)
                    w, h = rect[2], rect[3]
                    aspect_ratio = w / h if h > 0 else 0
                    if 2 < aspect_ratio < 10:  # Text-like aspect ratio
                        text_regions.append(rect)
            
            return text_regions
            
        except Exception as e:
            print(f"Error in text region detection: {e}")
            return []

    def _run_ocr(self, pil_image, cv_image):
        """Run OCR using the best available method"""
        if self.ocr_method == "florence":
            return self._run_florence_ocr(pil_image)
        elif self.ocr_method == "tesseract":
            return self._run_tesseract_ocr(pil_image)
        else:
            return self._run_basic_ocr(cv_image)

    def _clean_short_text(self, text):
        """
        Clean and optimize extracted short text
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
            
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common OCR artifacts and special characters
        cleaned = re.sub(r'[^\w\s\-.,!?()&\'\"]', '', cleaned)
        
        # Limit to max length for short text processing
        if len(cleaned) > self.max_text_length:
            cleaned = cleaned[:self.max_text_length].rsplit(' ', 1)[0] + "..."
            
        return cleaned

    def process_frame(self, frame):
        """
        Process frame to extract short text efficiently
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, extracted_text)
                - processed_frame (numpy.ndarray): Original frame with text overlay
                - extracted_text (str): Cleaned short text
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Convert OpenCV frame to PIL Image for advanced OCR methods
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Extract text using the best available OCR method
        extracted_text = self._run_ocr(pil_image, frame)
        
        # Add text overlay to output frame
        if extracted_text:
            # Create dark background for text overlay
            h, w = output.shape[:2]
            overlay = output.copy()
            
            # Calculate text box size
            text_lines = extracted_text.split('\n')
            max_line_length = max(len(line) for line in text_lines) if text_lines else 0
            text_height = len(text_lines) * 25 + 20
            text_width = min(max_line_length * 12 + 20, w - 20)
            
            # Draw background rectangle
            cv2.rectangle(overlay, (10, h - text_height - 10), (text_width, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
            
            # Add text lines
            y_position = h - text_height + 5
            for line in text_lines[:3]:  # Limit to 3 lines for short text
                if line.strip():
                    cv2.putText(
                        output,
                        line.strip(),
                        (15, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                    y_position += 25
            
            # Add OCR method indicator
            cv2.putText(
                output,
                f"OCR: {self.ocr_method}",
                (15, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1,
                cv2.LINE_AA
            )
        
        return output, extracted_text

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Short text processor doesn't process point clouds
        
        Args:
            point_cloud_data (Dict): Input point cloud data
            
        Returns:
            tuple: (point_cloud_data, message)
        """
        return point_cloud_data, {"message": "ShortTextProcessor does not process point cloud data."}


# Create processor instance
processor = ShortTextProcessor()
app = processor.app