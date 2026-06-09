import html
import re


def clean_tg_html(text: str) -> str:
    """
    Экранирует спецсимволы для Telegram HTML, сохраняя разрешенные теги.
    Разрешенные теги Telegram: <b>, <strong>, <i>, <em>, <u>, <ins>, <s>, <strike>, <del>, <blockquote>, <tg-spoiler>, <span class="tg-spoiler">, <a>, <code>, <pre>.
    """
    if not text:
        return ""

    # 1. Сначала экранируем вообще всё
    # Мы временно заменим существующие теги на плейсхолдеры, чтобы они не экранировались
    placeholders = []

    def tag_replacer(match):
        placeholders.append(match.group(0))
        return f"__TAG_PLACEHOLDER_{len(placeholders) - 1}__"

    # Ищем все разрешенные теги Telegram, включая теги с атрибутами
    tag_pattern = re.compile(
        r"<(?:/?(?:b|strong|i|em|u|ins|s|strike|del|tg-spoiler|blockquote|span|a|code|pre))(?:\s+[^>]*?)?>",
        re.IGNORECASE,
    )

    # Прячем теги
    text_with_placeholders = tag_pattern.sub(tag_replacer, text)

    # Экранируем всё остальное (&, <, >)
    unescaped_text = html.unescape(
        text_with_placeholders
    )  # Сначала убираем уже существующее (двойное экранирование плохо)
    escaped_text = html.escape(unescaped_text, quote=False)

    # Возвращаем теги обратно
    final_text = escaped_text
    for i, original_tag in enumerate(placeholders):
        # Если это ссылка <a>, принудительно экранируем & в href, так как TG этого требует
        if original_tag.lower().startswith("<a"):
            href_match = re.search(r'href=["\']([^"\']*)["\']', original_tag, re.IGNORECASE)
            if href_match:
                url = href_match.group(1)
                # Экранируем & в URL
                clean_url = url.replace("&", "&amp;")
                original_tag = original_tag.replace(url, clean_url)

        final_text = final_text.replace(f"__TAG_PLACEHOLDER_{i}__", original_tag)

    # 2. Балансировка тегов
    tags_to_balance = [
        "b",
        "strong",
        "i",
        "em",
        "u",
        "ins",
        "s",
        "strike",
        "del",
        "tg-spoiler",
        "blockquote",
        "span",
        "a",
        "code",
        "pre",
    ]

    for tag in tags_to_balance:
        open_pattern = re.compile(f"<{tag}\\b(?:\\s+[^>]*?)?>", re.IGNORECASE)
        close_pattern = re.compile(f"</{tag}>", re.IGNORECASE)

        # Удаляем "одинокие" закрывающие теги
        while True:
            opens = [m.start() for m in open_pattern.finditer(final_text)]
            closes = [m.start() for m in close_pattern.finditer(final_text)]

            # Если есть закрывающий тег до первого открывающего — он лишний
            found_bad_close = False
            for c in closes:
                # Если перед этим закрывающим нет ни одного неиспользованного открывающего
                if not any(o < c for o in opens):
                    # Удаляем этот конкретный закрывающий тег
                    final_text = final_text[:c] + final_text[c + len(f"</{tag}>") :]
                    found_bad_close = True
                    break

            if not found_bad_close:
                break

        # Дозакрываем открытые
        opens = len(open_pattern.findall(final_text))
        closes = len(close_pattern.findall(final_text))
        if opens > closes:
            final_text += f"</{tag}>" * (opens - closes)

    return final_text
