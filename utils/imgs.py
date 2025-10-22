from typing import Any
from PIL import Image
from io import BytesIO

def decode_dvbpr_image(img_field: Any) -> Image.Image:
    """
    Items[k]['imgs'] is bytes or a latin1 string of bytes. Return RGB PIL image.
    """
    if isinstance(img_field, str):
        img_bytes = img_field.encode("latin1")
    else:
        img_bytes = img_field
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return img
