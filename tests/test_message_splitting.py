import pytest
from src.interfaces.telegram_bot import split_html_message, split_plain_text
from src.interfaces.max_bot import _split_long_message

def test_telegram_split_html_message():
    assert split_html_message("<b>Hello World</b>", 20) == ["<b>Hello World</b>"]
    assert split_html_message("<b>Hello World</b>", 15) == ["<b>Hello</b>", "<b> World</b>"]
    assert split_html_message("<b>Hello\nWorld</b>", 15) == ["<b>Hello</b>", "<b>\nWorld</b>"]
    assert split_html_message("Some normal text", 10) == ["Some", " normal", " text"]
    assert split_html_message("<b>Nested <i>italic</i> bold</b>", 25) == ["<b>Nested <i>ital</i></b>", "<b><i>ic</i> bold</b>"]

def test_telegram_split_plain_text():
    assert split_plain_text("Some normal text", 10) == ["Some", " normal", " text"]
    assert split_plain_text("Short text", 20) == ["Short text"]

def test_max_split_long_message():
    assert _split_long_message("<b>Hello World</b>", 20) == ["<b>Hello World</b>"]
    assert _split_long_message("<b>Hello World</b>", 15) == ["<b>Hello</b>", "<b> World</b>"]
