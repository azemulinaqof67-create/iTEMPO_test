import pytest
from src.core.html_utils import clean_tg_html
from src.assistant.assistant import AssistantService

class DummyAssistant:
    _sanitize_response = AssistantService._sanitize_response
    _format_markdown_tables = AssistantService._format_markdown_tables
    _format_html_tables = AssistantService._format_html_tables

@pytest.fixture
def assistant():
    return DummyAssistant()

def test_clean_tg_html_basic():
    # Простая замена спецсимволов, игнорируя разрешенные теги
    assert clean_tg_html("Hello <b>World</b> & others") == "Hello <b>World</b> &amp; others"
    assert clean_tg_html("x < 5 and y > 10") == "x &lt; 5 and y &gt; 10"

def test_clean_tg_html_allowed_tags_with_attributes():
    # Теги с атрибутами не должны ломать балансировку
    assert clean_tg_html('<span class="tg-spoiler">Secret</span>') == '<span class="tg-spoiler">Secret</span>'
    assert clean_tg_html('<code class="language-python">x < 5</code>') == '<code class="language-python">x &lt; 5</code>'
    assert clean_tg_html('<a href="http://example.com?a=1&b=2">Link</a>') == '<a href="http://example.com?a=1&amp;b=2">Link</a>'

def test_clean_tg_html_balancing():
    # Балансировка тегов (удаление лишних закрывающих, дозакрытие открытых)
    assert clean_tg_html("<b>Hello") == "<b>Hello</b>"
    assert clean_tg_html("Hello </i>") == "Hello "
    assert clean_tg_html("<b>Hello <i>World</b>") == "<b>Hello <i>World</b></i>"

def test_sanitize_response_markdown_bold_italic(assistant):
    # Жирный и курсив
    assert assistant._sanitize_response("This is **bold** text") == "This is <b>bold</b> text"
    assert assistant._sanitize_response("This is *italic* text") == "This is <i>italic</i> text"
    assert assistant._sanitize_response("This is ***bold italic*** text") == "This is <b><i>bold italic</i></b> text"
    assert assistant._sanitize_response("This is __bold__ and _italic_") == "This is <b>bold</b> and <i>italic</i>"

def test_sanitize_response_markdown_links(assistant):
    # Ссылки
    assert assistant._sanitize_response("Check [this site](https://google.com)") == 'Check <a href="https://google.com">this site</a>'
    # Ссылка с & в URL
    assert assistant._sanitize_response("Check [link](https://site.com?a=1&b=2)") == 'Check <a href="https://site.com?a=1&amp;b=2">link</a>'

def test_sanitize_response_markdown_lists(assistant):
    # Списки
    markdown_list = "- Item 1\n- Item 2\n* Item 3"
    expected_list = "• Item 1\n• Item 2\n• Item 3"
    assert assistant._sanitize_response(markdown_list) == expected_list

def test_sanitize_response_markdown_headers(assistant):
    # Заголовки
    assert assistant._sanitize_response("### My Header") == "<b>My Header</b>"
    assert assistant._sanitize_response("## Header 2\nSome text") == "<b>Header 2</b>\n\nSome text"

def test_sanitize_response_markdown_blockquotes(assistant):
    # Цитаты
    quote_text = "> Line 1\n> Line 2\nNormal text"
    expected_quote = "<blockquote>Line 1\nLine 2</blockquote>\nNormal text"
    assert assistant._sanitize_response(quote_text) == expected_quote

def test_sanitize_response_code_blocks(assistant):
    # Блоки кода
    code_block = "```python\nif x < 5:\n    print('hello')\n```"
    expected_code = '<pre><code class="language-python">if x &lt; 5:\n    print(\'hello\')</code></pre>'
    assert assistant._sanitize_response(code_block) == expected_code

    # Блок кода без языка
    code_block_no_lang = "```\nsome raw code\n```"
    expected_code_no_lang = '<pre>some raw code</pre>'
    assert assistant._sanitize_response(code_block_no_lang) == expected_code_no_lang

    # Инлайн-код
    assert assistant._sanitize_response("Use `x < 5` to check") == "Use <code>x &lt; 5</code> to check"


def test_sanitize_response_html_tables(assistant):
    # Тест 2-колоночной HTML таблицы
    html_table_2col = (
        "<table>"
        "<tr><th>Имя</th><th>Телефон</th></tr>"
        "<tr><td>Девитьяров В.А.</td><td>2313</td></tr>"
        "</table>"
    )
    expected_2col = "• <b>Имя:</b> Девитьяров В.А. — <b>Телефон:</b> 2313"
    assert assistant._sanitize_response(html_table_2col) == expected_2col

    # Тест 3-колоночной HTML таблицы
    html_table_3col = (
        "<table>"
        "<tr><th>Имя</th><th>Роль</th><th>Кабинет</th></tr>"
        "<tr><td>Иван</td><td>Инженер</td><td>302</td></tr>"
        "</table>"
    )
    expected_3col = "• <b>Имя:</b> Иван\n  <b>Роль:</b> Инженер\n  <b>Кабинет:</b> 302"
    assert assistant._sanitize_response(html_table_3col) == expected_3col


def test_sanitize_response_markdown_tables(assistant):
    # Тест Markdown таблицы
    md_table = (
        "| Контакт | Телефон |\n"
        "|---------|---------|\n"
        "| Девитьяров | 2313 |\n"
        "| Андреев | 1888 |"
    )
    expected_md = (
        "• <b>Контакт:</b> Девитьяров — <b>Телефон:</b> 2313\n"
        "• <b>Контакт:</b> Андреев — <b>Телефон:</b> 1888"
    )
    assert assistant._sanitize_response(md_table) == expected_md

