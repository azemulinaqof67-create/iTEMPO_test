import re

def split_html_message(text: str, max_len: int = 4000) -> list[str]:
    """
    Разделяет HTML сообщение на чанки допустимого размера,
    сохраняя баланс HTML тегов.
    """
    if len(text) <= max_len:
        return [text]

    # Разделяем на теги и текст
    tokens = re.split(r'(<[^>]+>)', text)
    chunks = []
    
    current_chunk = []
    current_len = 0
    open_tags = []  # стек открытых тегов: список кортежей (имя_тега, полный_открывающий_тег)

    for token in tokens:
        if not token:
            continue
            
        if token.startswith('<') and token.endswith('>'):
            # Это тег
            tag_content = token[1:-1].strip()
            if tag_content.startswith('/'):
                # Закрывающий тег
                tag_name = tag_content[1:].strip().split()[0].lower()
                # Удаляем из стека последний открытый тег с таким именем
                for i in range(len(open_tags) - 1, -1, -1):
                    if open_tags[i][0] == tag_name:
                        open_tags.pop(i)
                        break
            elif not tag_content.endswith('/'):
                # Открывающий тег (игнорируем самозакрывающиеся теги)
                parts = tag_content.split()
                if parts:
                    tag_name = parts[0].lower()
                    if tag_name in ['b', 'strong', 'i', 'em', 'u', 'ins', 's', 'strike', 'del', 'span', 'tg-spoiler', 'a', 'code', 'pre', 'blockquote']:
                        open_tags.append((tag_name, token))
            
            # Проверяем, влезет ли этот тег
            if current_len + len(token) > max_len:
                # Нужно закрыть текущий чанк
                closing_tags = "".join(f"</{name}>" for name, _ in reversed(open_tags))
                current_chunk.append(closing_tags)
                chunks.append("".join(current_chunk))
                
                # Начинаем новый чанк и открываем теги заново
                current_chunk = []
                opening_tags = "".join(full_tag for _, full_tag in open_tags)
                current_chunk.append(opening_tags)
                current_len = len(opening_tags)
            
            current_chunk.append(token)
            current_len += len(token)
        else:
            # Это обычный текст. Он может быть длинным, поэтому его придется разбивать.
            text_fragment = token
            while text_fragment:
                # Сколько места осталось?
                # Учитываем закрывающие теги в конце чанка
                closing_tags_len = sum(len(name) + 3 for name, _ in open_tags)  # len('</name>')
                available = max_len - current_len - closing_tags_len
                
                if available <= 0:
                    closing_tags = "".join(f"</{name}>" for name, _ in reversed(open_tags))
                    current_chunk.append(closing_tags)
                    chunks.append("".join(current_chunk))
                    
                    current_chunk = []
                    opening_tags = "".join(full_tag for _, full_tag in open_tags)
                    current_chunk.append(opening_tags)
                    current_len = len(opening_tags)
                    available = max_len - current_len - closing_tags_len
                    if available <= 0:
                        available = 1
                
                if len(text_fragment) <= available:
                    current_chunk.append(text_fragment)
                    current_len += len(text_fragment)
                    break
                else:
                    sub_frag = text_fragment[:available]
                    split_idx = sub_frag.rfind('\n')
                    if split_idx <= 0:
                        split_idx = sub_frag.rfind(' ')
                    if split_idx <= 0:
                        split_idx = available
                    
                    part_to_add = text_fragment[:split_idx]
                    text_fragment = text_fragment[split_idx:]
                    
                    current_chunk.append(part_to_add)
                    
                    closing_tags = "".join(f"</{name}>" for name, _ in reversed(open_tags))
                    current_chunk.append(closing_tags)
                    chunks.append("".join(current_chunk))
                    
                    current_chunk = []
                    opening_tags = "".join(full_tag for _, full_tag in open_tags)
                    current_chunk.append(opening_tags)
                    current_len = len(opening_tags)

    if current_chunk:
        chunk_str = "".join(current_chunk)
        clean_text = re.sub(r'<[^>]+>', '', chunk_str).strip()
        if clean_text:
            chunks.append(chunk_str)
            
    return chunks

# Тесты
tests = [
    ("<b>Hello World</b>", 20, ["<b>Hello World</b>"]),
    ("<b>Hello World</b>", 15, ["<b>Hello</b>", "<b> World</b>"]),
    ("<b>Hello\nWorld</b>", 15, ["<b>Hello</b>", "<b>\nWorld</b>"]),
    ("Some normal text", 10, ["Some", " normal", " text"]),
    ("<b>Nested <i>italic</i> bold</b>", 25, ["<b>Nested <i>ital</i></b>", "<b><i>ic</i> bold</b>"])
]

for t, l, expected in tests:
    res = split_html_message(t, l)
    print(f"Text: {t!r}, len: {l}")
    print(f"Got: {res}")
    print(f"Expected: {expected}")
    assert res == expected, f"Failed: got {res} != expected {expected}"
    print("OK")
