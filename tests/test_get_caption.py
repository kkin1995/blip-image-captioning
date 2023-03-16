from image_captioning.get_caption import get_image_caption_from_model
from PIL import Image
import pytest


class TestGetImageCaptionFromModel:
    @pytest.fixture(scope="class")
    def path_to_image(self) -> str:
        return "tests/images/image_1.png"

    def test_string_output(self, path_to_image):
        image = Image.open(path_to_image)
        caption = get_image_caption_from_model(image, None)

        assert type(caption) == str

    def test_raised_warning(self, path_to_image):
        image = Image.open(path_to_image)
        with pytest.warns(UserWarning):
            _ = get_image_caption_from_model(image, None)

    def test_conditional_image_caption(self, path_to_image):
        image = Image.open(path_to_image)
        text = "A man running"
        caption = get_image_caption_from_model(image, None, text=text)
        index = caption.find(text.lower())

        assert index != -1
