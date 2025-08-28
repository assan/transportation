import pandas as pd
import logging
from datetime import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import re
from ortools.linear_solver import pywraplp
from collections import defaultdict

def setup_logging():
    """Настройка системы логирования"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/transport_optimization_{current_time}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def sanitize_filename(name):
    """Очистка имени файла от недопустимых символов"""
    invalid_chars = r'[<>:"/\\|?*]|\.\.+'
    sanitized = re.sub(invalid_chars, '_', name)
    sanitized = sanitized.strip().replace('__', '_')
    if not sanitized or sanitized.startswith('_'):
        sanitized = f"entity_{sanitized}"
    return sanitized

def load_data():
    """Загрузка всех необходимых данных"""
    logging.info("Загрузка входных данных...")

    factories = pd.read_csv('factories.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    factories.columns = ['factory']

    warehouses = pd.read_csv('warehouses.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    warehouses.columns = ['warehouse']

    products = pd.read_csv('products.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    products.columns = ['product']

    weeks = pd.read_csv('week.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    weeks.columns = ['week']
    weeks['week'] = weeks['week'].str.replace('W', '', regex=False)

    production = pd.read_csv('production.csv', skiprows=2, header=0, delimiter=';')
    production.columns = ['factory', 'product', 'week', 'amount']
    production['week'] = production['week'].str.replace('W', '', regex=False)
    production['amount'] = production['amount'].astype(float)

    demand = pd.read_csv('demand.csv', skiprows=2, header=0, delimiter=';')
    demand.columns = ['warehouse', 'week', 'demand']
    demand['week'] = demand['week'].str.replace('W', '', regex=False)
    demand['demand'] = demand['demand'].astype(float)
    valid_weeks = set(weeks['week'])
    demand = demand[demand['week'].isin(valid_weeks)]

    capacity = pd.read_csv('capacity.csv', skiprows=2, header=0, delimiter=';')
    capacity.columns = ['warehouse', 'week', 'capacity']
    capacity['week'] = capacity['week'].str.replace('W', '', regex=False)
    capacity['capacity'] = capacity['capacity'].astype(float)
    capacity = capacity[capacity['week'].isin(valid_weeks)]

    penalty = pd.read_csv('penalty.csv', skiprows=2, header=0, delimiter=';')
    penalty.columns = ['warehouse', 'penalty']
    penalty['penalty'] = penalty['penalty'].astype(float)

    costs = pd.read_csv('costs.csv', skiprows=2, header=0, delimiter=';')
    costs.columns = ['factory', 'warehouse', 'cost']
    costs['cost'] = costs['cost'].astype(float)

    for df, name in [(production, 'production'), (demand, 'demand'), (capacity, 'capacity'), (penalty, 'penalty'), (costs, 'costs')]:
        if df.isnull().any().any():
            raise ValueError(f"Обнаружены пропущенные значения в {name}.csv")
        if (df.select_dtypes(include=['float']).lt(0).any()).any():
            raise ValueError(f"Обнаружены отрицательные значения в {name}.csv")

    check_capacity(production, capacity, weeks)
    check_balance(production, demand, weeks)

    data = {
        'factories': factories['factory'].tolist(),
        'warehouses': warehouses['warehouse'].tolist(),
        'products': products['product'].tolist(),
        'weeks': sorted(weeks['week'].tolist(), key=int),
        'production': production.set_index(['factory', 'product', 'week'])['amount'].to_dict(),
        'demand': demand.set_index(['warehouse', 'week'])['demand'].to_dict(),
        'capacity': capacity.set_index(['warehouse', 'week'])['capacity'].to_dict(),
        'penalty': penalty.set_index('warehouse')['penalty'].to_dict(),
        'costs': costs.set_index(['factory', 'warehouse'])['cost'].to_dict()
    }

    if 'DUMMY' not in data['warehouses']:
        data['warehouses'].append('DUMMY')
        for w in data['weeks']:
            data['capacity'][('DUMMY', w)] = 1e12
            data['demand'][('DUMMY', w)] = 0
        data['penalty']['DUMMY'] = 0
        max_cost = max(data['costs'].values()) if data['costs'] else 1000
        for i in data['factories']:
            data['costs'][(i, 'DUMMY')] = max_cost
        logging.info("Добавлен фиктивный склад DUMMY для обработки излишков")

    for j in data['warehouses']:
        for w in data['weeks']:
            if (j, w) not in data['demand']:
                data['demand'][('j', 'w')] = 0

    demand_weeks = set(w for _, w in data['demand'].keys())
    capacity_weeks = set(w for _, w in data['capacity'].keys())
    valid_weeks = set(data['weeks'])
    if demand_weeks - valid_weeks:
        logging.warning(f"В demand.csv обнаружены недели, отсутствующие в week.csv: {demand_weeks - valid_weeks}")
    if capacity_weeks - valid_weeks:
        logging.warning(f"В capacity.csv обнаружены недели, отсутствующие в week.csv: {capacity_weeks - valid_weeks}")

    return data

def check_capacity(production_df, capacity_df, weeks_df):
    total_production = production_df.groupby('week')['amount'].sum()
    total_capacity = capacity_df.groupby('week')['capacity'].sum()
    for week in weeks_df['week']:
        prod = total_production.get(week, 0)
        cap = total_capacity.get(week, 0) + 1e12
        if prod > cap:
            logging.error(f"Неделя {week}: Производство {prod} > Вместимость {cap} (дефицит {prod - cap})")
        else:
            logging.info(f"Неделя {week}: Производство {prod} <= Вместимость {cap}")

def check_balance(production_df, demand_df, weeks_df):
    total_production = production_df.groupby('week')['amount'].sum()
    total_demand = demand_df.groupby('week')['demand'].sum()
    for week in weeks_df['week']:
        prod = total_production.get(week, 0)
        dem = total_demand.get(week, 0)
        if prod > dem:
            logging.info(f"Неделя {week}: Производство {prod} > Общий спрос {dem} (избыток {prod - dem})")
        elif prod < dem:
            logging.warning(f"Неделя {week}: Производство {prod} < Общий спрос {dem} (дефицит {dem - prod})")

def create_model(data):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Не удалось создать решатель SCIP")

    x_vals = {}
    stock = {}
    logging.info("Создание переменных...")
    for (i, j) in data['costs'].keys():
        for p in data['products']:
            for w in data['weeks']:
                x_vals[(i, j, p, w)] = solver.NumVar(0, solver.infinity(), f'x_{i}_{j}_{p}_{w}')
    for j in data['warehouses']:
        for p in data['products']:
            for w in data['weeks']:
                stock[(j, p, w)] = solver.NumVar(0, solver.infinity(), f'stock_{j}_{p}_{w}')

    logging.info("Формирование целевой функции...")
    objective = solver.Objective()
    for (i, j), cost in data['costs'].items():
        for p in data['products']:
            for w in data['weeks']:
                if (i, j, p, w) in x_vals:
                    objective.SetCoefficient(x_vals[(i, j, p, w)], cost)

    for j in data['warehouses']:
        if j in data['penalty']:
            for w in data['weeks']:
                if (j, w) in data['demand'] and data['demand'][(j, w)] > 0:
                    for i in data['factories']:
                        for p in data['products']:
                            if (i, j, p, w) in x_vals:
                                current_coeff = objective.GetCoefficient(x_vals[(i, j, p, w)])
                                objective.SetCoefficient(x_vals[(i, j, p, w)], current_coeff - data['penalty'][j])

    objective.SetMinimization()

    logging.info("Добавление ограничений...")
    # Ограничение на производство: вся продукция должна быть распределена
    for i in data['factories']:
        for p in data['products']:
            for w in data['weeks']:
                if (i, p, w) in data['production']:
                    constraint = solver.Constraint(data['production'][(i, p, w)], data['production'][(i, p, w)], f"Production_{i}_{p}_{w}")
                    for j in data['warehouses']:
                        if (i, j, p, w) in x_vals:
                            constraint.SetCoefficient(x_vals[(i, j, p, w)], 1)

    # Ограничение на вместимость: учитываем отгрузки и спрос
    for j in data['warehouses']:
        for w in data['weeks']:
            if (j, w) in data['capacity']:
                constraint = solver.Constraint(0, data['capacity'][(j, w)], f"Capacity_{j}_{w}")
                for p in data['products']:
                    if (j, p, w) in stock:
                        constraint.SetCoefficient(stock[(j, p, w)], 1)
                    for i in data['factories']:
                        if (i, j, p, w) in x_vals:
                            constraint.SetCoefficient(x_vals[(i, j, p, w)], 1)

    # Баланс запасов
    for j in data['warehouses']:
        for p in data['products']:
            for w in data['weeks']:
                week_idx = data['weeks'].index(w)
                constraint = solver.Constraint(0, 0, f"Stock_Balance_{j}_{p}_{w}")
                for i in data['factories']:
                    if (i, j, p, w) in x_vals:
                        constraint.SetCoefficient(x_vals[(i, j, p, w)], 1)
                if (j, p, w) in stock:
                    constraint.SetCoefficient(stock[(j, p, w)], -1)
                if week_idx > 0:
                    prev_w = data['weeks'][week_idx - 1]
                    if (j, p, prev_w) in stock:
                        constraint.SetCoefficient(stock[(j, p, prev_w)], 1)
                if (j, w) in data['demand'] and data['demand'][(j, w)] > 0:
                    constraint.SetCoefficient(stock[(j, p, w)], -data['demand'][(j, w)] / len(data['products']))

    logging.info("Решение модели...")
    solver.SetTimeLimit(7200 * 1000)
    solver.EnableOutput()

    start_time = time.time()
    status = solver.Solve()
    solve_time = time.time() - start_time

    u = {}
    v = {}
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        for w in data['weeks']:
            total_demand = sum(data['demand'].get((j, w), 0) for j in data['warehouses'] if (j, w) in data['demand'])
            total_load = sum(x_vals[(i, j, p, w)].solution_value() for i in data['factories'] for j in data['warehouses'] for p in data['products'] if (i, j, p, w) in x_vals)
            u[w] = max(0, total_demand - total_load)

        for i in data['factories']:
            for p in data['products']:
                for w in data['weeks']:
                    if (i, p, w) in data['production']:
                        sum_x = sum(x_vals[(i, j, p, w)].solution_value() for j in data['warehouses'] if (i, j, p, w) in x_vals)
                        v[(i, p, w)] = data['production'][(i, p, w)] - sum_x

    return solver, status, solve_time, x_vals, stock, u, v

def save_results(solver, status, solve_time, data, x_val, stock, u, v):
    status_dict = {
        pywraplp.Solver.OPTIMAL: 'OPTIMAL',
        pywraplp.Solver.FEASIBLE: 'FEASIBLE',
        pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
        pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
        pywraplp.Solver.ABNORMAL: 'ABNORMAL',
        pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED'
    }
    status_str = status_dict.get(status, 'UNKNOWN')

    logging.info(f"Статус решения: {status_str}")
    logging.info(f"Время решения: {solve_time:.2f} сек")

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        transport_plan = []
        for (i, j, p, w), var in x_val.items():
            if var.solution_value() > 1e-6:
                transport_plan.append({
                    'week': f"W{w}",
                    'factory': i,
                    'warehouse': j,
                    'product': p,
                    'amount': var.solution_value()
                })

        pd.DataFrame(transport_plan).to_csv('transport_plan_ortools.csv', index=False)
        logging.info(f"Сохранен план перевозок ({len(transport_plan)} записей)")

        stock_plan = []
        for (j, p, w), var in stock.items():
            if var.solution_value() > 1e-6:
                stock_plan.append({
                    'week': f"W{w}",
                    'warehouse': j,
                    'product': p,
                    'stock': var.solution_value()
                })

        if stock_plan:
            pd.DataFrame(stock_plan).to_csv('stock_plan_ortools.csv', index=False)
            logging.info(f"Сохранен план запасов ({len(stock_plan)} записей)")

        unsatisfied = []
        for w in data['weeks']:
            if w in u and u[w] > 1e-6:
                total_demand = sum(data['demand'].get((j, w), 0) for j in data['warehouses'])
                avg_penalty = sum(data['penalty'].get(j, 0) for j in data['warehouses']) / len(data['warehouses']) if data['warehouses'] else 0
                unsatisfied.append({
                    'week': f"W{w}",
                    'total_demand': total_demand,
                    'unsatisfied': u[w],
                    'penalty': u[w] * avg_penalty
                })

        total_unsatisfied = sum(u.values())
        if total_unsatisfied > 1e-6:
            pd.DataFrame(unsatisfied).to_csv('unsatisfied_demand_ortools.csv', index=False)
            logging.warning(f"Общий неудовлетворенный спрос: {total_unsatisfied:.2f}")

        unplaced = []
        for (i, p, w), value in v.items():
            if value > 1e-6:
                unplaced.append({
                    'factory': i,
                    'product': p,
                    'week': f"W{w}",
                    'production': data['production'][(i, p, w)],
                    'unplaced': value
                })

        total_unplaced = sum(v.values())
        if total_unplaced > 1e-6:
            pd.DataFrame(unplaced).to_csv('unplaced_production_ortools.csv', index=False)
            logging.error(f"Общий неразмещенный объем: {total_unplaced:.2f}")
        else:
            logging.info("Вся продукция успешно распределена по складам")

        transport_cost = 0.0
        for (i, j), cost in data['costs'].items():
            for p in data['products']:
                for w in data['weeks']:
                    if (i, j, p, w) in x_val:
                        transport_cost += cost * x_val[(i, j, p, w)].solution_value()

        unsatisfied_penalty = sum(item['penalty'] for item in unsatisfied)
        total_cost = transport_cost + unsatisfied_penalty

        logging.info(f"Транспортные затраты: {transport_cost:.2f}")
        logging.info(f"Штраф за неудовлетворенный спрос: {unsatisfied_penalty:.2f}")
        logging.info(f"Общие затраты: {total_cost:.2f}")

        total_production = sum(data['production'].values())
        total_demand = sum(data['demand'].values())
        total_placed = total_production - total_unplaced
        logging.info("\n=== Сводная статистика ===")
        logging.info(f"Всего произведено: {total_production:.2f}")
        logging.info(f"Всего размещено: {total_placed:.2f}")
        logging.info(f"Процент удовлетворенного спроса: {(1 - total_unsatisfied / total_demand if total_demand > 0 else 1) * 100:.2f}%")
        logging.info(f"Процент размещенной продукции: {(1 - total_unplaced / total_production if total_production > 0 else 1) * 100:.2f}%")

        if not os.path.exists('plots'):
            os.makedirs('plots')

        try:
            transport_df = pd.DataFrame(transport_plan)
            stock_df = pd.DataFrame(stock_plan)
            for j in data['warehouses']:
                if j == 'DUMMY':  # Пропускаем DUMMY склад
                    continue
                plt.figure(figsize=(12, 6))

                # Убедимся, что длина всех массивов одинаковая
                weeks_labels = [f"W{w}" for w in data['weeks']]
                x = np.arange(len(weeks_labels))

                # Вместимость
                capacity_data = [data['capacity'].get((j, w), 0) for w in data['weeks']]
                logging.info(f"Warehouse: {j}, Capacity data length: {len(capacity_data)}")
                if len(x) != len(capacity_data):
                    capacity_data = capacity_data[:len(x)]  # Подгоняем длину

                plt.plot(x, capacity_data, label='Capacity', color='black', linestyle='--', linewidth=2)

                # Запасы на начало недели с отладкой
                initial_stock_data = []
                for w in data['weeks']:
                    week_idx = data['weeks'].index(w)
                    if week_idx == 0:
                        initial_stock = 0  # Нет запасов на начало первой недели
                    else:
                        prev_w = data['weeks'][week_idx - 1]
                        initial_stock = 0
                        for p in data['products']:
                            key = (j, p, prev_w)
                            if key in stock:
                                initial_stock += stock[key].solution_value()
                        logging.info(f"Warehouse {j}, Week {prev_w} -> Initial Stock: {initial_stock}")
                    initial_stock_data.append(initial_stock)
                logging.info(f"Initial Stock data for {j}: {initial_stock_data}")

                plt.bar(x - 0.3, initial_stock_data, width=0.2, label='Initial Stock', color='#FF9999')

                # Отгруженный объем
                shipped_data = []
                for w in data['weeks']:
                    shipped = 0
                    for i in data['factories']:
                        for p in data['products']:
                            key = (i, j, p, w)
                            if key in x_val:
                                shipped += x_val[key].solution_value()
                    shipped_data.append(shipped)
                    logging.info(f"Warehouse {j}, Week {w}, Shipped: {shipped}")
                if len(x) != len(shipped_data):
                    shipped_data = shipped_data[:len(x)]  # Подгоняем длину

                plt.bar(x - 0.1, shipped_data, width=0.2, label='Shipped Volume', color='#66B2FF')

                # Спрос
                demand_data = [data['demand'].get((j, w), 0) for w in data['weeks']]
                if len(x) != len(demand_data):
                    demand_data = demand_data[:len(x)]  # Подгоняем длину

                plt.bar(x + 0.1, demand_data, width=0.2, label='Demand', color='#99FF99')

                # Проверка баланса запасов (для отладки)
                end_stock_data = []
                for w_idx, w in enumerate(data['weeks']):
                    if w_idx == 0:
                        end_stock = shipped_data[w_idx] - demand_data[w_idx] / len(data['products'])
                    else:
                        end_stock = initial_stock_data[w_idx] + shipped_data[w_idx] - demand_data[w_idx] / len(data['products'])
                    end_stock_data.append(max(0, end_stock))  # Ограничение неотрицательности
                    actual_stock = sum(stock.get((j, p, w), None).solution_value() if (j, p, w) in stock else 0 for p in
                                       data['products'])
                    logging.info(
                        f"Warehouse {j}, Week {w}, Calculated End Stock: {end_stock}, Actual Stock: {actual_stock}")

                plt.xlabel('Week')
                plt.ylabel('Volume')
                plt.title(f'Warehouse {j} Load: Capacity, Initial Stock, Shipped, and Demand')
                plt.xticks(x[::4], weeks_labels[::4], rotation=45)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                sanitized_file_name = sanitize_filename(j)
                plt.savefig(f'plots/warehouse_{sanitized_file_name}_load.png')
                plt.close()

            logging.info(f"Графики для складов сохранены в папку 'plots'.")

        except Exception as e:
            logging.error(f"Ошибка при создании графиков загруженности складов: {str(e)}")

    else:
        raise Exception(f"Не удалось найти решение. Статус: {status_str}")

def main():
    log_file = setup_logging()
    logging.info("=== Начало работы транспортного оптимизатора ===")

    try:
        data = load_data()
        solver, status, solve_time, x, stock, u, v = create_model(data)
        save_results(solver, status, solve_time, data, x, stock, u, v)
        logging.info("Оптимизация завершена успешно!")
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()