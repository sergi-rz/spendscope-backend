import base64

import pytest

from app.schemas import InputType
from app.services.preprocess import PreprocessError, _detect_kind, prepare

PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 16
PDF_MAGIC = b"%PDF-1.7\n" + b"rest"
XLSX_MAGIC = b"PK\x03\x04" + b"\x00" * 16


def test_text_passthrough():
    modality, payload = prepare(InputType.text, "MERCADONA -67,82", None)
    assert modality == "text"
    assert payload == "MERCADONA -67,82"


def test_png_image_becomes_vision_data_uri():
    content = base64.b64encode(PNG_MAGIC).decode()
    modality, payload = prepare(InputType.image, content, "receipt.jpg")
    assert modality == "vision"
    assert payload.startswith("data:image/png;base64,")


def test_jpeg_detected_by_magic_without_filename():
    content = base64.b64encode(JPEG_MAGIC).decode()
    modality, payload = prepare(InputType.image, content, None)
    assert modality == "vision"
    assert payload.startswith("data:image/jpeg;base64,")


def test_invalid_base64_raises():
    with pytest.raises(PreprocessError):
        prepare(InputType.image, "not!base64!!!", "x.jpg")


def test_detect_kind_pdf_by_magic():
    assert _detect_kind(PDF_MAGIC, "") == "pdf"


def test_detect_kind_excel_by_magic():
    assert _detect_kind(XLSX_MAGIC, "") == "excel"


def test_detect_kind_image_default():
    assert _detect_kind(PNG_MAGIC, "") == "image"
