from .base_processor import BaseProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional
from PIL import Image
import torch
import re

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
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Initialize Florence model for fast OCR
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype='auto'
            ).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.model_loaded = True
        except Exception as e:
            print(f"⚠️  Could not load model {model_id}: {e}")
            print("    Running in offline mode - processor will return placeholder text")
            self.model = None
            self.processor = None
            self.model_loaded = False
        
        print(f"✅ ShortTextProcessor initialized with {model_id} on {self.device}")

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
        """
        Run Florence OCR optimized for short text extraction
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            str: Extracted text
        """
        if not self.model_loaded:
            return "Short text processor (offline mode)"
            
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
                max_new_tokens=min(512, self.max_text_length + 50),  # Limit tokens for speed
                do_sample=False,
                num_beams=1,  # Reduce beams for speed
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
        
        # Convert OpenCV frame to PIL Image
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Extract text using Florence OCR
        extracted_text = self._run_florence_ocr(pil_image)
        
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