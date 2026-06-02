from src.core.config import Config
from src.assistant.assistant import AssistantService

def test():
    # Создаем фиктивный объект Config
    config = Config.from_env()
    assistant = AssistantService(config)
    
    test_cases = [
        # 1. Обычный список
        (
            "Вот список документов: <ul><li>Приказ о субботнике</li><li>График отпусков</li></ul>",
            "Вот список документов: \n• Приказ о субботнике\n• График отпусков"
        ),
        # 2. Абзацы и заголовки
        (
            "<h1>Заголовок</h1><p>Первый абзац.</p><p>Второй абзац.</p>",
            "<b>Заголовок</b>\n\nПервый абзац.\n\nВторой абзац."
        ),
        # 3. Таблица
        (
            "<table><tr><th>Имя</th><th>Телефон</th></tr><tr><td>Иван</td><td>123</td></tr></table>",
            "• <b>Имя:</b> Иван — <b>Телефон:</b> 123"
        ),
        # 4. Div и span
        (
            "<div><span>Некоторый текст</span> в контейнере.</div>",
            "Некоторый текст в контейнере."
        ),
        # 5. Лишние переносы строк
        (
            "Текст\n\n\n\nДругой текст",
            "Текст\n\nДругой текст"
        )
    ]
    
    success = True
    for i, (input_text, expected_output) in enumerate(test_cases):
        output = assistant._sanitize_response(input_text)
        if output.strip() == expected_output.strip():
            print(f"Test {i+1} PASSED")
        else:
            print(f"Test {i+1} FAILED")
            print(f"Input: {repr(input_text)}")
            print(f"Output: {repr(output)}")
            print(f"Expected: {repr(expected_output)}")
            print("-" * 40)
            success = False
            
    if success:
        print("All sanitization tests passed successfully!")

if __name__ == "__main__":
    test()
