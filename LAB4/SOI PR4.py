import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
import numpy as np

# === ЭТАП 1: Загрузка и первичный осмотр данных ===

# Указываем имя файла
file_path = 'FAOSTAT_data_ru_12-14-2025_obr.xlsx'

# Читаем файл один раз. Переменная df будет использоваться во всем коде дальше
df = pd.read_excel(file_path)

# Вывод первых 5 строк
print("--- Первые 5 строк ---")
print(df.head())

# Вывод последних 5 строк
print("\n--- Последние 5 строк ---")
print(df.tail())

# Проверка типов данных (чтобы убедиться, что Год и Значение - это числа)
print("\n--- Информация о типах данных ---")
print(df.info())

# === ЭТАП 2: Наглядное представление данных (Пункт 1.4) ===

# Настройка имени столбца (чтобы не менять везде вручную, если что)
val_col = 'Значение (kt)'

# Настройка размера шрифта
plt.rcParams.update({'font.size': 12})

# --- График 1: Линейная диаграмма (Динамика) ---
plt.figure(figsize=(12, 6))

plt.plot(df['Год'], df[val_col], color='#9370DB', linewidth=2.5, label='Выбросы CH4')

plt.fill_between(df['Год'], df[val_col], color='#E6E6FA', alpha=0.5)

# Добавляем точку максимума
max_val = df[val_col].max()
max_year = df.loc[df[val_col].idxmax(), 'Год']
plt.scatter(max_year, max_val, color='#4B0082', s=100, zorder=5) # Indigo цвет точки
plt.annotate(f'Максимум: {max_val:.2f} kt', xy=(max_year, max_val), xytext=(max_year + 3, max_val),
             arrowprops=dict(facecolor='#4B0082', shrink=0.05, alpha=0.7))

plt.title('Динамика выбросов метана от коневодства (1961–2023)', pad=20)
plt.xlabel('Год')
plt.ylabel('Выбросы (килотонны)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('graph_1_line_dynamics.png', dpi=300)
plt.show()


# --- График 2: Столбчатая диаграмма (Bar Chart) ---
plt.figure(figsize=(12, 6))

# Делаем градиент фиолетового (от светлого к темному в зависимости от высоты)
colors = plt.cm.Purples(df[val_col] / df[val_col].max() * 0.7 + 0.3)

plt.bar(df['Год'], df[val_col], color=colors, alpha=0.9)

plt.title('Объемы выбросов по годам (Столбчатая диаграмма)', pad=20)
plt.xlabel('Год')
plt.ylabel('Выбросы (килотонны)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('graph_2_bar_volumes.png', dpi=300)
plt.show()


# --- График 3: Гистограмма распределения (Histogram) ---
plt.figure(figsize=(10, 6))

plt.hist(df[val_col], bins=15, color='#9370DB', edgecolor='#663399', alpha=0.8)

plt.title('Гистограмма распределения уровней выбросов', pad=20)
plt.xlabel('Уровень выбросов (kt)')
plt.ylabel('Частота (кол-во лет)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('graph_3_hist_distribution.png', dpi=300)
plt.show()

print("Графики сохранены в папку с проектом.")

# === ЭТАП 3: Проверка на стационарность (Пункт 1.5) ===

print(">>> ------------------- <<<")

# 1. Подготовка данных для линейной регрессии (чтобы посчитать остатки)
X = df[['Год']].values
y = df[val_col].values

# Строим простую линейную модель
model = LinearRegression()
model.fit(X, y)
trend_values = model.predict(X)
residuals = y - trend_values  # Остатки = Факт - Тренд

# 2. Расчет статистики Дарбина-Уотсона
dw_stat = durbin_watson(residuals)
print(f"Статистика Дарбина-Уотсона: {dw_stat:.4f}")

# 3. Построение ACF и PACF (Коррелограммы)
# Рассчитываем значения функций
lag_count = 15  # Количество лагов (сдвигов)
acf_vals = acf(residuals, nlags=lag_count)
pacf_vals = pacf(residuals, nlags=lag_count)
lags = range(len(acf_vals))

# Настраиваем график (2 подграфика: сверху ACF, снизу PACF)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# --- График ACF ---
# Рисуем "ножки" (stem plot)
markerline, stemlines, baseline = ax1.stem(lags, acf_vals, basefmt=" ")
plt.setp(stemlines, 'color', '#9370DB', 'linewidth', 2) # Фиолетовые линии
plt.setp(markerline, 'markerfacecolor', '#4B0082', 'markeredgecolor', '#4B0082') # Темно-фиолетовые точки
plt.setp(baseline, 'color', 'black', 'linewidth', 1)

# Доверительный интервал
conf_level = 1.96 / np.sqrt(len(df))
ax1.fill_between(lags, -conf_level, conf_level, color='#E6E6FA', alpha=0.5)
ax1.set_title('Автокорреляционная функция (ACF) исходного ряда', fontsize=14)
ax1.set_ylabel('Коэффициент корреляции')
ax1.grid(True, linestyle='--', alpha=0.6)

# --- График PACF ---
markerline, stemlines, baseline = ax2.stem(lags, pacf_vals, basefmt=" ")
plt.setp(stemlines, 'color', '#9370DB', 'linewidth', 2)
plt.setp(markerline, 'markerfacecolor', '#4B0082', 'markeredgecolor', '#4B0082')
plt.setp(baseline, 'color', 'black', 'linewidth', 1)

# Доверительный интервал
ax2.fill_between(lags, -conf_level, conf_level, color='#E6E6FA', alpha=0.5)
ax2.set_title('Частичная автокорреляционная функция (PACF) исходного ряда', fontsize=14)
ax2.set_xlabel('Лаг (сдвиг в годах)')
ax2.set_ylabel('Коэффициент корреляции')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('graph_4_correlogram.png', dpi=300)
plt.show()

# Дарбин-Уотсон для ИСХОДНОГО ряда (сырые данные)
dw_raw = durbin_watson(df[val_col])

# Дарбин-Уотсон для ОСТАТКОВ линейной модели
dw_resid = durbin_watson(residuals)

print(f"Число 1 (Для исходного ряда): {dw_raw:.4f}")
print(f"Число 2 (Для остатков модели): {dw_resid:.4f}")
print(">>> ------------------- <<<")

# === ЭТАП 4: Выделение тренда и цикличности (Пункт 1.6) ===

print("\n--- ЭТАП 4: Декомпозиция ряда ---")

# 1. Выделение ТРЕНДА методом скользящего среднего (Moving Average)
# Используем окно = 3 года, чтобы сгладить мелкие скачки
window_size = 3
df['Trend_MA'] = df[val_col].rolling(window=window_size, center=True).mean()

# 2. Выделение ЦИКЛИЧЕСКОЙ составляющей (Cycle)
# Формула: Цикл = Факт - Тренд
df['Cycle'] = df[val_col] - df['Trend_MA']

# 3. Визуализация (2 подграфика)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- Подграфик 1: Исходный ряд и Тренд ---
# Исходный ряд (точки и тонкая линия)
ax1.plot(df['Год'], df[val_col], color='#DDA0DD', marker='o', markersize=4, linestyle='-', alpha=0.6, label='Исходный ряд')
# Тренд (Жирная темная линия)
ax1.plot(df['Год'], df['Trend_MA'], color='#4B0082', linewidth=3, label=f'Тренд (SMA, окно={window_size})')

ax1.set_title(f'Выделение тренда (Скользящее среднее)', fontsize=14)
ax1.set_ylabel('Выбросы (kt)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# --- Подграфик 2: Циклическая составляющая ---
# Рисуем линию вокруг нуля
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5) # Линия нуля
ax2.plot(df['Год'], df['Cycle'], color='#8A2BE2', linewidth=2, marker='o', markersize=4, label='Циклическая составляющая')

ax2.fill_between(df['Год'], df['Cycle'], 0, where=(df['Cycle'] >= 0), color='#E6E6FA', interpolate=True, alpha=0.6)
ax2.fill_between(df['Год'], df['Cycle'], 0, where=(df['Cycle'] <= 0), color='#D8BFD8', interpolate=True, alpha=0.6)

ax2.set_title('Циклическая составляющая (Отклонение от тренда)', fontsize=14)
ax2.set_xlabel('Год')
ax2.set_ylabel('Отклонение (kt)')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('graph_5_decomposition.png', dpi=300)
plt.show()

print("Декомпозиция выполнена. Графики сохранены.")

# === ЭТАП 5: Выявление аномалий (Метод Ирвина) ===
print("\n--- ЭТАП 5: Метод Ирвина ---")

# 1. Расчет характеристик ряда
y = df[val_col].values
n = len(y)
mean_y = np.mean(y)
std_y = np.std(y, ddof=1) # Несмещенная оценка

print(f"Количество наблюдений (n): {n}")
print(f"Среднее значение: {mean_y:.4f} kt")
print(f"Стандартное отклонение (sigma_y): {std_y:.4f} kt")

# 2. Расчет коэффициентов Ирвина для каждого года
# lambda_t = |y_t - y_{t-1}| / sigma_y
irwin_values = []
years = df['Год'].values

print("\nТоп-5 самых высоких значений критерия Ирвина:")
for t in range(1, n):
    diff = abs(y[t] - y[t-1])
    lambda_t = diff / std_y
    irwin_values.append((years[t], lambda_t, diff))

# Сортируем, чтобы найти самые большие скачки
irwin_values.sort(key=lambda x: x[1], reverse=True)

for year, lam, diff in irwin_values[:5]:
    print(f"Год: {year}, Изменение: {diff:.4f}, Lambda: {lam:.4f}")

# Критическое значение для n=60-70 при alpha=0.05 составляет примерно 1.3 - 1.5
# (В таблицах Ирвина для n=60 критическое значение ~1.04, но часто берут 1.5 для надежности)
crit_val = 1.3
max_lambda = irwin_values[0][1]

if max_lambda > crit_val:
    print(f"\nВЫВОД: Обнаружена аномалия в {irwin_values[0][0]} году!")
else:
    print(f"\nВЫВОД: Аномалий, превышающих 1.3 sigma, не обнаружено.")

# === ЭТАП 6: Проверка наличия тренда и сглаживание (Пункт 1.8) ===
from scipy import stats

print("\n--- ЭТАП 6: Сравнение средних и сглаживание ---")

# 1. Разбиение ряда на две части
n = len(df)
n1 = n // 2  # Первая половина
n2 = n - n1  # Вторая половина

part1 = df.iloc[:n1]
part2 = df.iloc[n1:]

# 2. Расчет статистик (Среднее и Дисперсия)
mean1 = part1[val_col].mean()
var1 = part1[val_col].var(ddof=1) # s^2_1
mean2 = part2[val_col].mean()
var2 = part2[val_col].var(ddof=1) # s^2_2

# 3. Расчет t-критерия (Формула для неравных дисперсий со Слайда 7)
# t = |y1 - y2| / sqrt(s1^2/n1 + s2^2/n2)
t_stat = abs(mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))

# Критическое значение t-Стьюдента (alpha=0.05, df approx n-2)
t_crit = stats.t.ppf(1 - 0.05/2, n - 2)

print(f"Период 1: {part1['Год'].min()}-{part1['Год'].max()}")
print(f"Среднее 1: {mean1:.4f}, Дисперсия 1: {var1:.4f}")
print(f"Период 2: {part2['Год'].min()}-{part2['Год'].max()}")
print(f"Среднее 2: {mean2:.4f}, Дисперсия 2: {var2:.4f}")
print(f"t-статистика: {t_stat:.4f}")
print(f"Критическое значение t (при a=0.05): {t_crit:.4f}")

# 4. Визуализация (Сглаживание)
# Скользящее среднее (окно 3) уже посчитано как 'Trend_MA' в прошлом шаге.

plt.figure(figsize=(12, 6))

# 1. Исходный ряд:
# - linestyle='--' (штриховая линия, чтобы отличалась от сплошного тренда)
# - marker='o' (кружочки, чтобы видеть конкретные годы)
# - alpha=0.7 (делаем более заметным, меньше прозрачности)
plt.plot(df['Год'], df[val_col],
         color='#9370DB',       # MediumPurple (светлее тренда, но яркий)
         linestyle='--',        # Штриховая линия
         linewidth=1,           # Тонкая
         marker='o',            # Маркер-кружок
         markersize=4,
         alpha=0.7,             # Прозрачность всего 30%
         label='Исходный ряд')

# 2. Сглаженный ряд (Тренд):
# - Оставляем сплошным и жирным
plt.plot(df['Год'], df['Trend_MA'],
         color='#4B0082',       # Indigo (Темно-фиолетовый)
         linewidth=3,           # Жирная линия
         label='Сглаженный ряд (SMA, m=3)')

plt.title('Сглаживание временного ряда методом скользящей средней', pad=20)
plt.xlabel('Год')
plt.ylabel('Выбросы (kt)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('graph_7_trend_smoothing.png', dpi=300)
plt.show()

# === ЭТАП 7: Построение тренда (Полином 3-го порядка - Слайд 11) ===
from sklearn.metrics import r2_score

print("\n--- ЭТАП 7: Полиномиальная модель (3-го порядка) ---")

# 1. Подготовка данных
t = np.arange(1, len(df) + 1)
y_real = df[val_col].values

# 2. Расчет коэффициентов полинома (3-й порядок: y = a0 + a1*t + a2*t^2 + a3*t^3)
# np.polyfit возвращает коэффициенты от старшей степени к младшей (a3, a2, a1, a0)
coeffs = np.polyfit(t, y_real, 3)
p = np.poly1d(coeffs) # Создаем функцию-полином

# Распаковываем коэффициенты для отчета (в обратном порядке, чтобы a0 был свободным членом)
# polyfit дает: [a3, a2, a1, a0] -> мы хотим a0, a1, a2, a3
a3, a2, a1, a0 = coeffs

print(f"Параметры модели y = a0 + a1*t + a2*t^2 + a3*t^3:")
print(f"a0 (Free): {a0:.4f}")
print(f"a1 (t):    {a1:.4f}")
print(f"a2 (t^2):  {a2:.4f}")
print(f"a3 (t^3):  {a3:.6f}")

# 3. Расчет модельных значений
y_pred = p(t)

# 4. Расчет R^2
r2 = r2_score(y_real, y_pred)
print(f"Коэффициент детерминации R^2: {r2:.4f}")

# 5. Визуализация
plt.figure(figsize=(12, 6))

# Исходный ряд
plt.plot(df['Год'], df[val_col],
         color='#9370DB', linestyle='--', marker='o', markersize=4, alpha=0.6,
         label='Исходный ряд')

# Модель
plt.plot(df['Год'], y_pred,
         color='#DC143C', linewidth=3,
         label=f'Полином 3-го порядка ($R^2={r2:.2f}$)')

plt.title('Полиномиальный тренд выбросов метана', pad=20)
plt.xlabel('Год')
plt.ylabel('Выбросы (kt)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('graph_8_poly_trend.png', dpi=300)
plt.show()

# === ЭТАП 8: Оценка остаточной компоненты (Анализ остатков) ===
import math

print("\n--- ЭТАП 8: Анализ остатков ---")

# 1. Расчет остатков
residuals = y_real - y_pred

# 2. Формирование последовательности знаков (по Медиане)
median_res = np.median(residuals)
signs = []
for r in residuals:
    if r > median_res:
        signs.append('+')
    elif r < median_res:
        signs.append('-')
    # Если равно, пропускаем (стандартная практика)

n = len(signs) # Общее число наблюдений (после исключения равных медиане)

# 3. Подсчет серий (N) и максимальной длины серии (K)
N_runs = 1       # Количество серий (N на слайде)
max_run_len = 0  # Длина самой большой серии (K на слайде)
current_run_len = 1

for i in range(1, n):
    if signs[i] == signs[i-1]:
        current_run_len += 1
    else:
        # Серия закончилась, проверяем длину
        if current_run_len > max_run_len:
            max_run_len = current_run_len
        current_run_len = 1
        N_runs += 1

# Проверка последней серии
if current_run_len > max_run_len:
    max_run_len = current_run_len

# 4. Расчет критических значений (Формулы со слайда 16)
# K < [3.3 * (lg(n) + 1)]
# log10 в питоне - это math.log10
k_threshold = int(3.3 * (math.log10(n) + 1))

# N > [0.5 * (n + 1 - 1.96 * sqrt(n-1))]
n_threshold = int(0.5 * (n + 1 - 1.96 * math.sqrt(n - 1)))

print(f"Общее число наблюдений (n): {n}")
print(f"Количество серий (N): {N_runs}")
print(f"Длина самой большой серии (K): {max_run_len}")
print("-" * 20)
print(f"Критическое значение для K (должно быть меньше): {k_threshold}")
print(f"Критическое значение для N (должно быть больше): {n_threshold}")

# 5. Проверка гипотезы H0
condition_1 = max_run_len < k_threshold
condition_2 = N_runs > n_threshold

if condition_1 and condition_2:
    print("\nВЫВОД: Гипотеза H0 принимается (Остатки случайны).")
else:
    print("\nВЫВОД: Гипотеза H0 отвергается (Остатки НЕ случайны).")
    if not condition_1: print(f"Причина: Слишком длинная серия (K={max_run_len} >= {k_threshold})")
    if not condition_2: print(f"Причина: Слишком мало серий (N={N_runs} <= {n_threshold})")

# Визуализация остатков
plt.figure(figsize=(12, 6))

# Линия нуля
plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# График остатков
plt.plot(df['Год'], residuals,
         color='#9370DB',
         marker='o', markersize=4, linestyle='-', linewidth=1.5,
         label='Остаточная компонента')

plt.title('Диаграмма остаточной компоненты', pad=20)
plt.xlabel('Год')
plt.ylabel('Отклонение (kt)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig('graph_9_residuals.png', dpi=300)
plt.show()

# === ЭТАП 9: Сравнительный анализ методов сглаживания ===
print("\n--- ЭТАП 9: Сравнение методов сглаживания (SMA vs EMA) ---")

# 1. Расчет Экспоненциального сглаживания (EMA)
# alpha = 0.3 (как в задании)
alpha = 0.3
df['EMA'] = df[val_col].ewm(alpha=alpha, adjust=False).mean()

# SMA (Скользящее среднее) у нас уже есть в колонке 'Trend_MA' (окно=3)

# 2. Визуализация сравнения
plt.figure(figsize=(12, 7))

# Исходный ряд (фиолетовый, на фоне)
plt.plot(df['Год'], df[val_col],
         color='#9370DB',       # Светло-серый
         marker='o', markersize=3,
         linestyle='-', linewidth=1,
         label='Исходный ряд')

# Скользящее среднее (SMA) - Красный
plt.plot(df['Год'], df['Trend_MA'],
         color='#DC143C',       # Crimson
         linewidth=2,
         label='Скользящее среднее (SMA, m=3)')

# Экспоненциальное сглаживание (EMA) - Зеленый
plt.plot(df['Год'], df['EMA'],
         color='#228B22',       # ForestGreen
         linewidth=2,
         label=f'Экспоненциальное сглаживание (alpha={alpha})')

plt.title('Сравнение методов сглаживания: SMA vs EMA', pad=20)
plt.xlabel('Год')
plt.ylabel('Выбросы (kt)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('graph_11_smoothing_comparison.png', dpi=300)
plt.show()

print("График сравнения методов сглаживания построен.")


# === ЭТАП 10: Дополнительные критерии (Тест Дикки-Фуллера) ===
from statsmodels.tsa.stattools import adfuller

print("\n--- ЭТАП 10: Расширенный тест Дикки-Фуллера (ADF) ---")

# Выполняем тест на исходном ряде
adf_result = adfuller(df[val_col])

adf_stat = adf_result[0]
p_value = adf_result[1]
crit_values = adf_result[4]

print(f"ADF-статистика: {adf_stat:.4f}")
print(f"p-значение: {p_value:.4f}")
print("Критические значения:")
for key, value in crit_values.items():
    print(f"   {key}: {value:.4f}")

print("-" * 30)

# Интерпретация
# H0: Ряд нестационарен (есть единичный корень)
# H1: Ряд стационарен

if p_value < 0.05:
    print("ВЫВОД: p-value < 0.05. Нулевая гипотеза отвергается.")
    print("Ряд является СТАЦИОНАРНЫМ (что маловероятно для нашего графика).")
else:
    print("ВЫВОД: p-value > 0.05. Недостаточно оснований отвергнуть нулевую гипотезу.")
    print("Ряд является НЕСТАЦИОНАРНЫМ (подтверждает наличие тренда).")

    # Дополнительная проверка: Сравнение статистики с критическими значениями
    # Если статистика БОЛЬШЕ (правее) критического значения (например, -1.5 > -3.5), то H0 не отвергается
    if adf_stat > crit_values['5%']:
        print(f"Подтверждение: Статистика ({adf_stat:.4f}) выше критического уровня 5% ({crit_values['5%']:.4f}).")
