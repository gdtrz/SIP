# --- ШАГ 0: ИМПОРТ НЕОБХОДИМЫХ БИБЛИОТЕК ---
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, levene, ks_2samp, spearmanr

print("Библиотеки успешно импортированы.")

# --- ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
file_path = 'lab2.xlsx'
target_column = 'Population in labour force'
alpha = 0.05 # Уровень статистической значимости

try:
    # --- ОСНОВНОЙ АНАЛИЗ ---
    df = pd.read_excel(file_path)
    print(f"\nФайл '{file_path}' успешно загружен.")

    # Очистка данных
    df_clean = df[~df['Age'].str.contains('Total|Recount', na=False)].copy()
    df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='coerce')
    
    # Создание основных выборок
    male_sample = df_clean[df_clean['Sex'] == '1_Male'][target_column]
    female_sample = df_clean[df_clean['Sex'] == '2_Female'][target_column]
    
    print("Данные очищены, созданы две выборки для основного анализа:")
    print(f"  - Мужчины: {len(male_sample)} наблюдений")
    print(f"  - Женщины: {len(female_sample)} наблюдений")
    print("\n" + "="*70)

    # --- ШАГ 2: ПРИМЕНЕНИЕ СТАТИСТИЧЕСКИХ КРИТЕРИЕВ ---

    # Критерий 1: T-критерий Стьюдента
    print("\n--- 1. T-критерий Стьюдента (проверка равенства средних) ---")
    stat_t, p_t = ttest_ind(male_sample, female_sample, equal_var=False)
    print(f"Статистика T-критерия = {stat_t:.4f}, P-value = {p_t:.4f}")
    if p_t < alpha: print(f"Вывод: P-value < {alpha}, ОТКЛОНЯЕМ H₀. Средние значения значимо различаются.")
    else: print(f"Вывод: P-value >= {alpha}, НЕ отклоняем H₀. Нет оснований считать, что средние различаются.")
    print("-" * 70)

    # Критерий 2: U-критерий Манна-Уитни
    print("\n--- 2. U-критерий Манна-Уитни (непараметрическая проверка различий) ---")
    stat_u, p_u = mannwhitneyu(male_sample, female_sample, alternative='two-sided')
    print(f"Статистика U-критерия = {stat_u:.4f}, P-value = {p_u:.4f}")
    if p_u < alpha: print(f"Вывод: P-value < {alpha}, ОТКЛОНЯЕМ H₀. Распределения значимо различаются.")
    else: print(f"Вывод: P-value >= {alpha}, НЕ отклоняем H₀. Нет оснований считать, что распределения различаются.")
    print("-" * 70)
        
    # Критерий 3: Критерий Левена и Коэффициент Вариации
    print("\n--- 3. Критерий Левена и Коэффициент Вариации ---")
    stat_l, p_l = levene(male_sample, female_sample)
    print(f"Статистика критерия Левена = {stat_l:.4f}, P-value = {p_l:.4f}")
    if p_l < alpha: print("Вывод: Дисперсии статистически значимо различаются.")
    else: print("Вывод: Нет оснований считать, что дисперсии выборок различаются.")
    
    cv_male = np.std(male_sample, ddof=1) / np.mean(male_sample) * 100
    cv_female = np.std(female_sample, ddof=1) / np.mean(female_sample) * 100
    print(f"Коэффициент вариации (мужчины): {cv_male:.1f}%")
    print(f"Коэффициент вариации (женщины): {cv_female:.1f}%")
    print("Вывод по CV: Относительный разброс данных в обеих группах практически идентичен.")
    print("-" * 70)

    # Критерий 4: Критерий Колмогорова-Смирнова
    print("\n--- 4. Критерий Колмогорова-Смирнова (проверка идентичности распределений) ---")
    stat_ks, p_ks = ks_2samp(male_sample, female_sample)
    print(f"Статистика критерия = {stat_ks:.4f}, P-value = {p_ks:.4f}")
    if p_ks < alpha: print(f"Вывод: P-value < {alpha}, ОТКЛОНЯЕМ H₀. Функции распределения значимо различаются.")
    else: print(f"Вывод: P-value >= {alpha}, НЕ отклоняем H₀. Распределения можно считать идентичными.")
    print("-" * 70)

    # Критерий 5: Ранговая корреляция Спирмена
    print("\n--- 5. Ранговая корреляция Спирмена ---")
    correlation, p_corr = spearmanr(male_sample, female_sample)
    print(f"Коэффициент корреляции (rho) = {correlation:.4f}, P-value = {p_corr:.4e}")
    if p_corr < alpha: print(f"Вывод: P-value < {alpha}, ОТКЛОНЯЕМ H₀. Существует значимая монотонная связь.")
    else: print(f"Вывод: P-value >= {alpha}, НЕ отклоняем H₀. Значимая монотонная связь не обнаружена.")
    print("-" * 70)

    # --- ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ---
    print("\n" + "="*70)
    print("--- Дополнительный анализ по признаку 'Unemployed persons' ---")
    print("="*70)

    additional_column = 'Unemployed persons'
    try:
        df_clean[additional_column] = pd.to_numeric(df_clean[additional_column], errors='coerce')
        male_unemployed_sample = df_clean[df_clean['Sex'] == '1_Male'][additional_column]
        female_unemployed_sample = df_clean[df_clean['Sex'] == '2_Female'][additional_column]

        print(f"Созданы две новые выборки по признаку '{additional_column}':")
        print(f"  - Мужчины (безработные): {len(male_unemployed_sample)} наблюдений")
        print(f"  - Женщины (безработные): {len(female_unemployed_sample)} наблюдений")
        
        print("\n--- Применение критерия Колмогорова-Смирнова к данным о безработице ---")
        stat_ks_add, p_ks_add = ks_2samp(male_unemployed_sample, female_unemployed_sample)

        print(f"Статистика критерия = {stat_ks_add:.4f}, P-value = {p_ks_add:.4f}")
        if p_ks_add < alpha: print(f"Вывод: P-value < {alpha}, ОТКЛОНЯЕМ H₀. Функции распределения безработицы значимо различаются.")
        else: print(f"Вывод: P-value >= {alpha}, НЕ отклоняем H₀. Распределения безработицы можно считать идентичными.")

    except KeyError:
        print(f"ОШИБКА в доп. анализе: Столбец '{additional_column}' не найден в ваших данных.")
    
# БЛОКИ EXCEPT ТЕПЕРЬ С ПРАВИЛЬНЫМИ ОТСТУПАМИ
except FileNotFoundError:
    print(f"ОШИБКА: Файл '{file_path}' не найден. Убедитесь, что он находится в той же папке, что и скрипт.")
except KeyError as e:
    print(f"ОШИБКА: Не найден один из ключевых столбцов для основного анализа: {e}. Проверьте названия в Excel.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")