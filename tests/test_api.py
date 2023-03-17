import pytest
import requests


class TestApi:
    @pytest.fixture(scope="class")
    def url(self) -> str:
        return "http://127.0.0.1:8000"

    @pytest.fixture(scope="class")
    def path_to_image(self) -> str:
        return "tests/images/image_1.png"

    def test_return_data_type(self, url: str, path_to_image: str):
        image = {"image": open(path_to_image, "rb")}
        text = "a man"
        to_send = {"text": text}
        res = requests.post(url=url, data=to_send, files=image)
        res = res.json()
        assert isinstance(res, dict)
        assert type(res["captions"]) == list
