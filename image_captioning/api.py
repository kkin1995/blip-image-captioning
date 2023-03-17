from fastapi import FastAPI, UploadFile, File
from get_caption import get_image_caption_from_model
from PIL import Image
import io
import requests

app = FastAPI()


@app.post("/")
async def image_caption(image: UploadFile = File(...), text: list[str] = None):
    image = await image.read()
    image = Image.open(io.BytesIO(image))

    caption = get_image_caption_from_model(image, text=text)

    return {"captions": caption}
