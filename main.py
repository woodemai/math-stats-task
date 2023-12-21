# Подключаем библиотеки
import  os
import scipy
import pandas as pd
import numpy as np
from numpy import arange
from scipy import stats
import plotly.express as px
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sb
from tabulate import tabulate


# загружаем новые файлы
def read_csv_files(folder_path):
    data_frames = {}

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder '{folder_path}' does not exist.")

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file into a DataFrame
            try:
                df = pd.read_csv(file_path, sep=';')
                data_frames[file_name] = df
            except Exception as e:
                print(f"Error reading file '{file_name}': {e}")

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
        prices = df['Price']

        # Рассчитываем ежедневные доходности
        returns = prices.pct_change()

        # Рассчитываем среднюю доходность и риск
        average_return = np.mean(returns)
        risk = np.std(returns)

        results[stock_name] = {'Average Return': average_return, 'Risk': risk}

    return results

returns_risk = calculate_returns_risk_for_multiple_stocks(stocks.copy())

for stock_name, metrics in returns_risk.items():
    print(f"\nМетрики для акции '{stock_name}':")
    print(f"Средняя доходность: {metrics['Average Return']:.4f}")
    print(f"Риск: {metrics['Risk']:.4f}")


# 3.3 Построить матрицы корреляций и ковариаций
def calculate_covariance_correlation_matrices(data_frames):
    # Объединяем данные по столбцу 'Date'
    merged_data = pd.concat([df['Price'] for df in data_frames.values()], axis=1, keys=data_frames.keys())

    # Рассчитываем матрицу ковариаций
    covariance_matrix = merged_data.pct_change().cov()

    # Рассчитываем матрицу корреляций
    correlation_matrix = merged_data.pct_change().corr()
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
returns = merged_data.pct_change()
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

# Построение эффективного фронта
def plot_efficient_frontier(results):
    # Построение эффективного фронта
    plt.figure(figsize=(12, 8))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(label='Шарп-отношение')
    plt.title('Эффективный фронт')
    plt.xlabel('Стандартное отклонение (риск)')
    plt.ylabel('Ожидаемая доходность')
    plt.show()


plot_efficient_frontier(results)