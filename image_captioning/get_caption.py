from __future__ import annotations
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
import matplotlib.pyplot as plt
import warnings


def get_image_caption_from_model(
    image: Image.Image,
    model_name: str | None = "Salesforce/blip-image-captioning-base",
    text: str | list | None = None,
    plot_image: bool = False,
) -> str:
    """
    Accepts a pre-trained image captioning model and an image to return a caption.

    Parameters:
    -----------
    image (PIL.Image.Image): Image to be captioned.
    model_name (str): String name of the pre-trained image captioning model. Currently accepts only "Salesforce/blip-image-captioning-base".
    text (str): String to be used for conditional captioning. Prompt for captioning model.
    plot_image (bool): Flag to plot the input image.

    Returns:
    -----------
    caption (str): Generated caption of the image.
    """

    if model_name == None:
        warnings.warn(
            "Warning: No model name provided. Using Salesforce/blip-image-captioning-base"
        )
        model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    image.convert("RGB")
    if plot_image:
        image_array = np.array(image)
        plt.imshow(image_array)

    if text != None:
        # conditional image captioning
        inputs = processor(image, text, return_tensors="pt", padding=True)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(out, skip_special_tokens=True)
    else:
        # unconditional image captioning
        inputs = processor(image, return_tensors="pt", padding=True)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.batch_decode(out, skip_special_tokens=True)

    return caption
