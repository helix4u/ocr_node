from typing import Literal, Optional
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter, ImageOps
from invokeai.app.invocations.primitives import ColorField, ImageField, ImageOutput, IntegerOutput
from pydantic import Field
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    UIComponent,
    UIType,
    WithMetadata,
    WithWorkflow,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import StringOutput
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

@invocation(
    "ocr_node",
    title="Image to Text",
    tags=["image", "text", "OCR"],
    category="image",
    version="1.0.0",
)
class ImageToTextInvocation(BaseInvocation):
    """Extract text from an image using OCR."""
    
    # Inputs
    image: ImageField = InputField(default=None, description="The image for text extraction")

    def invoke(self, context: InvocationContext) -> StringOutput:
        # Get PIL image
        image = context.services.images.get_pil_image(self.image.image_name)
        
        # Perform OCR to extract text
        extracted_text = pytesseract.image_to_string(image)
        
        return StringOutput(value=extracted_text)

@invocation_output("ocr_node_output")
class OCRNodeInvocationOutput(BaseInvocationOutput):
    extracted_text: str = OutputField(description="The Output")