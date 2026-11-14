import io

import pytest

from api import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def test_health(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}


def test_sentiment_endpoint(client):
    r = client.post("/api/sentiment", json={"text": "What a fantastic, wonderful product!"})
    assert r.status_code == 200 and r.get_json()["label"] == "Positive"


def test_missing_text_is_400(client):
    r = client.post("/api/sentiment", json={})
    assert r.status_code == 400 and "error" in r.get_json()


def test_empty_text_is_400(client):
    assert client.post("/api/sentiment", json={"text": "  "}).status_code == 400


def test_transcribe_requires_file(client):
    assert client.post("/api/transcribe").status_code == 400


def test_transcribe_rejects_bad_extension(client):
    r = client.post("/api/transcribe", data={"file": (io.BytesIO(b"x"), "notes.pdf")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "Unsupported" in r.get_json()["error"]


def test_transcribe_rejects_bad_model(client):
    r = client.post("/api/transcribe", data={"file": (io.BytesIO(b"x"), "a.wav"), "model": "huge"},
                    content_type="multipart/form-data")
    assert r.status_code == 400
