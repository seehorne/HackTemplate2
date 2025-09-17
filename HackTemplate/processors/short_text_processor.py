from .base_processor import BaseProcessor
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import cv2
from typing import Dict, Union, Tuple, Optional
from PIL import Image
import torch

class ShortTextProcessor(BaseProcessor):
    def __init__(self,
                 model_id='microsoft/Florence-2-large',
                 use_gpu=True,
                 confidence_threshold=0.3):
        """
        Initialize Short Text processor - mimics SeeingAI's Short Text feature
        
        Args:
            model_id (str): Florence model identifier
            use_gpu (bool): Whether to use GPU acceleration
            confidence_threshold (float): Minimum confidence for text detection
        """
        super().__init__()
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype='auto'
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.confidence_threshold = confidence_threshold

    def _extract_text_from_image(self, image):
        """
        Extract text from the image using Florence-2 OCR
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            str: Extracted text content
        """
        try:
            task_prompt = '<OCR>'
            
            # Prepare inputs
            inputs = self.processor(
                text=task_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device, torch.float16)

            # Generate results
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

            # Process output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Parse the result
            result = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            
            # Extract just the text content
            if result and task_prompt in result:
                ocr_result = result[task_prompt]
                if 'text' in ocr_result and isinstance(ocr_result['text'], list):
                    # Join all detected text pieces
                    extracted_text = ' '.join(ocr_result['text'])
                    return extracted_text.strip()
                elif isinstance(ocr_result, str):
                    return ocr_result.strip()
            
            return ""
            
        except Exception as e:
            # Log the error but return empty string to avoid breaking the processor
            print(f"Error in OCR processing: {str(e)}")
            return ""

    def process_frame(self, frame):
        """
        Process frame to extract short text - similar to SeeingAI Short Text
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (None, extracted_text) - no visual output, just text
        """
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Extract text
            extracted_text = self._extract_text_from_image(pil_image)
            
            # If no text found, return appropriate message
            if not extracted_text:
                extracted_text = "No text detected in image"
            
            # Return None for visual output (like SeeingAI, this is audio-focused)
            # and the extracted text as the result
            return None, extracted_text
            
        except Exception as e:
            # Handle any errors gracefully
            error_message = f"Error processing image: {str(e)}"
            return None, error_message
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not supported for text extraction
        """
        return point_cloud_data, {"message": "ShortTextProcessor does not process point cloud data."}


processor = ShortTextProcessor()
app = processor.app