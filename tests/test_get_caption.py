from image_captioning.get_caption import get_image_caption_from_model
from PIL import Image
import pytest


class TestGetImageCaptionFromModel:
    @pytest.fixture(scope="class")
    def path_to_image(self) -> str:
        return "tests/images/image_1.png"

    def test_string_output(self, path_to_image):
        image = Image.open(path_to_image)
        caption = get_image_caption_from_model(image)

        assert type(caption) == list
        assert type(caption[0]) == str

    def test_conditional_image_caption(self, path_to_image):
        image = Image.open(path_to_image)
        text = "A man running"
        caption = get_image_caption_from_model(image, text=text)
        index = caption[0].find(text.lower())

        assert index != -1

    def test_conditional_image_caption_with_list(self, path_to_image):
        image = Image.open(path_to_image)
        text = ["A man", "A guy", "Football"]
        caption = get_image_caption_from_model(image, text=text)

        assert type(caption) == list
        assert len(caption) == 3
