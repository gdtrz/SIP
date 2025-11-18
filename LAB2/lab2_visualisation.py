import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Цветовая палитра ---
MALE_COLOR = '#7fc97f'  # Зеленый
FEMALE_COLOR = '#bca0dc' # Светло-фиолетовый

file_path = 'lab2.xlsx' 

try:
    df = pd.read_excel(file_path)
    print("Файл успешно загружен. Начинаем подготовку данных для визуализации...")

    # Очистка данных
    if 'Age' in df.columns and df['Age'].dtype == 'object':
         df_clean = df[~df['Age'].str.contains('Total|Recount', na=False)].copy()
    else:
         df_clean = df.copy()
         
    df_clean = df_clean[df_clean['Sex'].isin(['1_Male', '2_Female'])]
    
    # Убедимся, что наш целевой столбец имеет числовой тип
    df_clean['Population in labour force'] = pd.to_numeric(df_clean['Population in labour force'], errors='coerce')
    print("Подготовка данных завершена.")

    # --- Визуализация 1: ЛИНЕЙНЫЙ ГРАФИК для 'Population in labour force' ---
    male_data = df_clean[df_clean['Sex'] == '1_Male']
    female_data = df_clean[df_clean['Sex'] == '2_Female']
    
    ages = pd.to_numeric(male_data['Age'].str.extract(r'(\d+)')[0], errors='coerce')

    fig1, ax1 = plt.subplots(figsize=(15, 8)) # Делаем график шире
    
    ax1.plot(ages, male_data['Population in labour force'], label='Мужчины (1_Male)', color=MALE_COLOR, marker='o', markersize=3, linewidth=2)
    ax1.plot(ages, female_data['Population in labour force'], label='Женщины (2_Female)', color=FEMALE_COLOR, marker='o', markersize=3, linewidth=2)

    ax1.set_ylabel('Количество человек', fontsize=12)
    ax1.set_xlabel('Возраст (лет)', fontsize=12)
    ax1.set_title('Динамика населения в составе рабочей силы по полу и возрасту', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Форматируем ось Y для читаемости больших чисел
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y/1000)}K')) # Показываем в тысячах (K)
    
    # Устанавливаем метки на оси X каждые 5 лет для читаемости
    ax1.set_xticks(np.arange(min(ages), max(ages)+1, 5))
    plt.xticks(rotation=45)

    fig1.tight_layout()
    plt.savefig('linechart_labour_force.png')
    print("Линейный график сохранен как linechart_labour_force.png")

    # --- Визуализация 2: Коробчатая диаграмма ---
    custom_palette = {'1_Male': MALE_COLOR, '2_Female': FEMALE_COLOR}
    
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    
    sns.boxplot(x='Sex', y='Population in labour force', data=df_clean, ax=ax2, palette=custom_palette)
    
    ax2.set_title('Распределение населения в составе рабочей силы по полу', fontsize=16)
    ax2.set_xlabel('Пол', fontsize=12)
    ax2.set_ylabel('Количество человек', fontsize=12)
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: f'{int(y/1000)}K'))
            
    plt.tight_layout()
    plt.savefig('boxplot_labour_force.png')
    print("Коробчатая диаграмма сохранена как boxplot_labour_force.png")
    
    plt.show()

except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден.")
except Exception as e:
    print(f"Произошла ошибка: {e}")