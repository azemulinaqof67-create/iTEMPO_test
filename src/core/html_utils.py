import re
import html

def clean_tg_html(text: str) -> str:
    """
    Экранирует спецсимволы для Telegram HTML, сохраняя разрешенные теги.
    Разрешенные теги Telegram: <b>, <i>, <a>, <code>, <pre>.
    """
    if not text:
        return ""

    # 1. Сначала экранируем вообще всё
    # Мы временно заменим существующие теги на плейсхолдеры, чтобы они не экранировались
    placeholders = []
    
    def tag_replacer(match):
        placeholders.append(match.group(0))
        return f"__TAG_PLACEHOLDER_{len(placeholders)-1}__"

    # Ищем теги: <b>, </b>, <i>, </i>, <a href="...">, </a>, <code>, </code>, <pre>, </pre>
    tag_pattern = re.compile(r'<(?:/?(?:b|i|code|pre)|a(?:\s+href=["\'][^"\']*["\'])?\s*/?)>', re.IGNORECASE)
    
    # Прячем теги
    text_with_placeholders = tag_pattern.sub(tag_replacer, text)
    
    # Экранируем всё остальное (&, <, >)
    unescaped_text = html.unescape(text_with_placeholders) # Сначала убираем уже существующее (двойное экранирование плохо)
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
                clean_url = url.replace('&', '&amp;')
                original_tag = original_tag.replace(url, clean_url)
        
        final_text = final_text.replace(f"__TAG_PLACEHOLDER_{i}__", original_tag)

    # 2. Балансировка тегов
    for tag in ['b', 'i', 'code', 'pre']:
        # Удаляем "одинокие" закрывающие теги
        while True:
            # Находим все позиции открывающих и закрывающих тегов
            opens = [m.start() for m in re.finditer(f'<{tag}>', final_text, re.IGNORECASE)]
            closes = [m.start() for m in re.finditer(f'</{tag}>', final_text, re.IGNORECASE)]
            
            # Если есть закрывающий тег до первого открывающего — он лишний
            found_bad_close = False
            for c in closes:
                # Если перед этим закрывающим нет ни одного неиспользованного открывающего
                if not any(o < c for o in opens):
                    # Удаляем этот конкретный закрывающий тег
                    final_text = final_text[:c] + final_text[c + len(f'</{tag}>'):]
                    found_bad_close = True
                    break
            
            if not found_bad_close:
                break

        # Дозакрываем открытые
        opens = len(re.findall(f'<{tag}>', final_text, re.IGNORECASE))
        closes = len(re.findall(f'</{tag}>', final_text, re.IGNORECASE))
        if opens > closes:
            final_text += f"</{tag}>" * (opens - closes)
            
    # Аналогично для <a>
    while True:
        opens = [m.start() for m in re.finditer(r'<a\s+href', final_text, re.IGNORECASE)]
        closes = [m.start() for m in re.finditer(r'</a>', final_text, re.IGNORECASE)]
        found_bad_close = False
        for c in closes:
            if not any(o < c for o in opens):
                final_text = final_text[:c] + final_text[c + 4:]
                found_bad_close = True
                break
        if not found_bad_close:
            break

    a_opens = len(re.findall(r'<a\s+href', final_text, re.IGNORECASE))
    a_closes = len(re.findall(r'</a>', final_text, re.IGNORECASE))
    if a_opens > a_closes:
        final_text += "</a>" * (a_opens - a_closes)

    return final_text
