from typing import Optional
from PIL import Image
import pytesseract

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import StringOutput

# Update this path if your Tesseract installation is elsewhere
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@invocation(
    "ocr_node",
    title="Image to Text",
    tags=["image", "text", "OCR"],
    category="image",
    version="1.0.1",
)
class ImageToTextInvocation(BaseInvocation):
    """Extract text from an image using OCR."""

    image: ImageField = InputField(default=None, description="The image for text extraction")

    def invoke(self, context: InvocationContext) -> StringOutput:
        # Retrieve the PIL image
        image = context.images.get_pil(self.image.image_name)

        # Perform OCR to extract text
        extracted_text = pytesseract.image_to_string(image)

        return StringOutput(value=extracted_text)
