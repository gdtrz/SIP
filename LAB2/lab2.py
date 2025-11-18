import pandas as pd

file_path = 'lab2.xlsx'

try:
    df = pd.read_excel(file_path)

    # Выводим первые 5 строк набора данных
    print("--- Первые 5 записей набора данных ---")
    print(df.head())

    # Добавляем разделитель для наглядности
    print("\n" + "="*50 + "\n")

    # Выводим последние 5 строк набора данных
    print("--- Последние 5 записей набора данных ---")
    print(df.tail())

except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден.")