# Подключаем библиотеки
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# загружаем все файлы
def read_csv_files(folder_path):
    data_frames = {}
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep=';')
            data_frames[os.path.basename(file_path)] = df
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    return data_frames


def print_csv_data(data_frames):
    for file_name, df in data_frames.items():
        print(f"Contents of '{file_name}':\n")
        print(df)
        print("\n" + "="*50 + "\n")  # Separating each DataFrame with a line


folder = './resources'
stocks = read_csv_files(folder)
print(print_csv_data(stocks.copy()))

# 3. Работа с начальными данными

# 3.1 Вывести график изменения акций
def plot_prices(data_frames):
    # Создайте пустой график
    plt.figure(figsize=(10, 6))

    # Переберите каждый файл
    for file_name, df in data_frames.items():
        # Преобразуйте столбец "Date" в формат даты, если он не является датой
        df['Date'] = pd.to_datetime(df['Date'])

        # Постройте график для столбца "High"
        plt.plot(df['Date'], df['Price'], label=file_name.split(".")[0])

    # Настройте оси и заголовок
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Prices Over Time')

    # Добавьте легенду
    plt.legend()

    # Показать график
    plt.show()


plot_prices(stocks.copy())


# 3.2 Рассчитать среднюю доходность и риск для отдельной акции
def calculate_returns_risk_for_multiple_stocks(data_frames):
    results = {}

    for stock_name, df in data_frames.items():
        returns = df['Price'].pct_change()
        average_return = returns.mean()
        risk = returns.std()

        results[stock_name] = {'Average Return': average_return, 'Risk': risk}

    return results


returns_risk = calculate_returns_risk_for_multiple_stocks(stocks.copy())

for stock_name, metrics in returns_risk.items():
    print(f"\nМетрики для акции '{stock_name}':")
    print(f"Средняя доходность: {metrics['Average Return']:.4f}")
    print(f"Риск: {metrics['Risk']:.4f}")


# 3.3 Построить матрицы корреляций и ковариаций
def calculate_covariance_correlation_matrices(data_frames):
    prices = pd.concat([df['Price'] for df in data_frames.values()], axis=1, keys=data_frames.keys())
    returns = prices.pct_change(fill_method=None)
    covariance_matrix = returns.cov()
    correlation_matrix = returns.corr()

    return covariance_matrix, correlation_matrix


def plot_matrix(matrix, title, cmap='OrRd'):
    # Визуализируем матрицу
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap=cmap, interpolation='none')
    for i in range(len(matrix.columns)):
        for j in range(len(matrix.columns)):
            plt.text(j, i, f'{matrix.iloc[i, j]:.4f}', ha='center', va='center')
    plt.colorbar(label='Value')
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    plt.title(title)
    plt.show()


covariance, correlation = calculate_covariance_correlation_matrices(stocks.copy())
plot_matrix(covariance, "Ковариации")
plot_matrix(correlation, "Корреляции", cmap='YlGnBu')


# 4. Построение модели Марковица и проверка модели на данных, посмотреть на разной ожидаемой доходности
merged_data = pd.concat([df['Price'] for df in stocks.values()], axis=1, keys=stocks.keys())
returns = merged_data.pct_change(fill_method=None)
expected_returns = returns.mean()
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    # Ожидаемая доходность портфеля
    portfolio_return = np.sum(weights * expected_returns) * 252  # Умножаем на 252 рабочих дня в году

    # Стандартное отклонение портфеля (риск)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)

    # Записываем результаты в массив
    results[0,i] = portfolio_return
    results[1,i] = portfolio_volatility
    results[2,i] = portfolio_return / portfolio_volatility  # Отношение Шарпа (показатель эффективности)


# Добавляем вывод диаграммы для максимальной и минимальной доходности
def plot_optimal_portfolios(results):
    # Находим портфели с максимальной и минимальной доходностью
    max_return_portfolio = results[:, results[0, :].argmax()]
    min_return_portfolio = results[:, results[0, :].argmin()]

    # Построение графика
    plt.figure(figsize=(12, 8))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Шарп-отношение')
    plt.title('Эффективный фронт с оптимальными портфелями')
    plt.xlabel('Стандартное отклонение (риск)')
    plt.ylabel('Ожидаемая доходность')

    # Выделение точек с максимальной и минимальной доходностью
    plt.scatter(max_return_portfolio[1], max_return_portfolio[0], marker='*', color='red', s=500, label='Максимальная доходность')
    plt.scatter(min_return_portfolio[1], min_return_portfolio[0], marker='*', color='blue', s=500, label='Минимальная доходность')

    plt.legend()

    plt.show()


plot_optimal_portfolios(results)

# Круговая диаграмма для оптимального портфеля с максимальной доходностью
def plot_max_return_pie(weights, max_return_portfolio):
    labels = stocks.keys()
    sizes = weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Оптимальный портфель для максимальной доходности')
    plt.show()

# Круговая диаграмма для оптимального портфеля с минимальной доходностью
def plot_min_return_pie(weights, min_return_portfolio):
    labels = stocks.keys()
    sizes = weights
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Оптимальный портфель для минимальной доходности')
    plt.show()

# Создаем случайные веса для оптимальных портфелей
weights_max_return = np.random.random(len(stocks))
weights_max_return /= np.sum(weights_max_return)

weights_min_return = np.random.random(len(stocks))
weights_min_return /= np.sum(weights_min_return)

# Оптимальные портфели
max_return_portfolio = results[:, results[0, :].argmax()]
min_return_portfolio = results[:, results[0, :].argmin()]

# Выводим результаты
plot_max_return_pie(weights_max_return, max_return_portfolio)
plot_min_return_pie(weights_min_return, min_return_portfolio)
