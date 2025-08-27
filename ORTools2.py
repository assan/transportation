import pandas as pd
import logging
from datetime import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import re
from ortools.linear_solver import pywraplp

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
            data['capacity'][('DUMMY', w)] = 1e9
            data['demand'][('DUMMY', w)] = 0
        data['penalty']['DUMMY'] = 0
        max_cost = max(data['costs'].values()) if data['costs'] else 1000
        for i in data['factories']:
            data['costs'][(i, 'DUMMY')] = max_cost * 10
        logging.info("Добавлен фиктивный склад DUMMY для обработки неразмещённой продукции")

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
        cap = total_capacity.get(week, 0)
        if prod > cap:
            logging.warning(f"Неделя {week}: Производство {prod} > Вместимость {cap} (дефицит {prod - cap})")

def check_balance(production_df, demand_df, weeks_df):
    total_production = production_df.groupby('week')['amount'].sum()
    total_demand = demand_df.groupby('week')['demand'].sum()
    for week in weeks_df['week']:
        prod = total_production.get(week, 0)
        dem = total_demand.get(week, 0)
        if prod > dem:
            logging.warning(f"Неделя {week}: Производство {prod} > Общий спрос {dem} (избыток {prod - dem})")
        elif prod < dem:
            logging.warning(f"Неделя {week}: Производство {prod} < Общий спрос {dem} (дефицит {dem - prod})")

def create_model(data):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Не удалось создать решатель SCIP")

    x = {}
    logging.info("Создание переменных...")
    for (i, j) in data['costs'].keys():
        for p in data['products']:
            for w in data['weeks']:
                x[(i, j, p, w)] = solver.NumVar(0, solver.infinity(), f'x_{i}_{j}_{p}_{w}')

    logging.info("Формирование целевой функции...")
    objective = solver.Objective()
    for (i, j), cost in data['costs'].items():
        for p in data['products']:
            for w in data['weeks']:
                if (i, j, p, w) in x:
                    objective.SetCoefficient(x[(i, j, p, w)], cost)

    for j in data['warehouses']:
        if j in data['penalty']:
            for w in data['weeks']:
                if (j, w) in data['demand']:
                    for i in data['factories']:
                        for p in data['products']:
                            if (i, j, p, w) in x:
                                current_coeff = objective.GetCoefficient(x[(i, j, p, w)])
                                objective.SetCoefficient(x[(i, j, p, w)], current_coeff - data['penalty'][j])

    penalty_unplaced = max(data['penalty'].values()) * 1000 if data['penalty'] else 1000
    for i in data['factories']:
        for p in data['products']:
            for w in data['weeks']:
                if (i, p, w) in data['production']:
                    for j in data['warehouses']:
                        if (i, j, p, w) in x:
                            current_coeff = objective.GetCoefficient(x[(i, j, p, w)])
                            objective.SetCoefficient(x[(i, j, p, w)], current_coeff - penalty_unplaced)

    objective.SetMinimization()
    full_production=0
    logging.info("Добавление ограничений...")
    for i in data['factories']:
        for p in data['products']:
            for w in data['weeks']:
                if (i, p, w) in data['production']:
                    constraint = solver.Constraint(0, data['production'][(i, p, w)], f"Production_{i}_{p}_{w}")
                    full_production+=data['production'][(i, p, w)]
                    for j in data['warehouses']:
                        if (i, j, p, w) in x:
                            constraint.SetCoefficient(x[(i, j, p, w)], 1)

    for j in data['warehouses']:
        for w in data['weeks']:
            if (j, w) in data['capacity']:
                constraint = solver.Constraint(0, data['capacity'][(j, w)], f"Capacity_{j}_{w}")
                for i in data['factories']:
                    for p in data['products']:
                        if (i, j, p, w) in x:
                            constraint.SetCoefficient(x[(i, j, p, w)], 1)
    full_demand=0
    for w in data['weeks']:
        total_demand = sum(data['demand'].get((j, w), 0) for j in data['warehouses'] if (j, w) in data['demand'])
        full_demand+=total_demand
        if total_demand > 0:
            constraint = solver.Constraint(0, total_demand, f"Total_Demand_{w}")
            for j in data['warehouses']:
                for i in data['factories']:
                    for p in data['products']:
                        if (i, j, p, w) in x:
                            constraint.SetCoefficient(x[(i, j, p, w)], 1)
    logging.info(f"Общий спрос: {full_demand}")
    logging.info(f'Общее производство {full_production}')
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
            total_load = sum(x[(i, j, p, w)].solution_value() for i in data['factories'] for j in data['warehouses'] for p in data['products'] if (i, j, p, w) in x)
            u[w] = max(0, total_demand - total_load)

        for i in data['factories']:
            for p in data['products']:
                for w in data['weeks']:
                    if (i, p, w) in data['production']:
                        sum_x = sum(x[(i, j, p, w)].solution_value() for j in data['warehouses'] if (i, j, p, w) in x)
                        v[(i, p, w)] = data['production'][(i, p, w)] - sum_x

    return solver, status, solve_time, x, u, v, penalty_unplaced

def save_results(solver, status, solve_time, data, x, u, v, penalty_unplaced):
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
        for (i, j, p, w), var in x.items():
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
                    'unplaced': value,
                    'penalty': value * penalty_unplaced
                })

        total_unplaced = sum(v.values())
        if total_unplaced > 1e-6:
            pd.DataFrame(unplaced).to_csv('unplaced_production_ortools.csv', index=False)
            logging.error(f"Общий неразмещенный объем: {total_unplaced:.2f}")

        transport_cost = 0.0
        for (i, j), cost in data['costs'].items():
            for p in data['products']:
                for w in data['weeks']:
                    if (i, j, p, w) in x:
                        transport_cost += cost * x[(i, j, p, w)].solution_value()

        unsatisfied_penalty = sum(item['penalty'] for item in unsatisfied)
        unplaced_penalty = sum(item['penalty'] for item in unplaced)
        total_cost = transport_cost + unsatisfied_penalty + unplaced_penalty

        logging.info(f"Транспортные затраты: {transport_cost:.2f}")
        logging.info(f"Штраф за неудовлетворенный спрос: {unsatisfied_penalty:.2f}")
        logging.info(f"Штраф за неразмещенную продукцию: {unplaced_penalty:.2f}")
        logging.info(f"Общие затраты: {total_cost:.2f}")

        total_production = sum(data['production'].values())
        total_demand = sum(data['demand'].values())
        total_placed = total_production - total_unplaced
        logging.info("\n=== Сводная статистика ===")
        logging.info(f"Всего произведено: {total_production:.2f}")
        logging.info(f"Всего размещено: {total_placed:.2f}")
        logging.info(f"Остатки на фабриках: {total_unplaced:.2f}")
        logging.info(f"Процент удовлетворенного спроса: {(1 - total_unsatisfied / total_demand if total_demand > 0 else 1) * 100:.2f}%")
        logging.info(f"Процент размещенной продукции: {(1 - total_unplaced / total_production if total_production > 0 else 1) * 100:.2f}%")

        if not os.path.exists('plots'):
            os.makedirs('plots')

        try:
            weeks_labels = [f"W{w}" for w in data['weeks']]
            total_demand_data = [sum(data['demand'].get((j, w), 0) for j in data['warehouses']) for w in data['weeks']]
            total_load_data = [sum(x[(i, j, p, w)].solution_value() for i in data['factories'] for j in data['warehouses'] for p in data['products'] if (i, j, p, w) in x) for w in data['weeks']]
            total_production_data = [sum(data['production'].get((i, p, w), 0) for i in data['factories'] for p in data['products']) for w in data['weeks']]

            logging.info(f"График спроса, удовлетворенного спроса и производства: недели={weeks_labels}, len(недели)={len(weeks_labels)}, total_demand_data={total_demand_data}, len(total_demand_data)={len(total_demand_data)}, total_load_data={total_load_data}, len(total_load_data)={len(total_load_data)}, total_production_data={total_production_data}, len(total_production_data)={len(total_production_data)}")

            if len(total_demand_data) != len(weeks_labels) or len(total_load_data) != len(weeks_labels) or len(total_production_data) != len(weeks_labels):
                logging.error(f"Несоответствие размеров для графика: len(weeks)={len(weeks_labels)}, len(total_demand_data)={len(total_demand_data)}, len(total_load_data)={len(total_load_data)}, len(total_production_data)={len(total_production_data)}")
            else:
                plt.figure(figsize=(12, 6))
                x = np.arange(len(weeks_labels))
                width = 0.3
                plt.bar(x - width, total_demand_data, width, label='Total Demand', color='#FF00FF')
                plt.bar(x, total_load_data, width, label='Satisfied Demand', color='#36A2EB')
                plt.bar(x + width, total_production_data, width, label='Total Production', color='#4BC0C0')
                plt.xlabel('Week')
                plt.ylabel('Volume')
                plt.title('Total Demand, Satisfied Demand, and Production by Week')
                plt.xticks(x[::4], weeks_labels[::4], rotation=45)
                plt.legend()
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                try:
                    plt.savefig('plots/total_demand_satisfied_production.png')
                    logging.info("Сохранён график спроса, удовлетворенного спроса и производства: plots/total_demand_satisfied_production.png")
                except Exception as e:
                    logging.error(f"Ошибка при сохранении графика спроса, удовлетворенного спроса и производства: {str(e)}")
                plt.close()
        except Exception as e:
            logging.error(f"Ошибка при создании графика спроса, удовлетворенного спроса и производства: {str(e)}")

    else:
        raise Exception(f"Не удалось найти решение. Статус: {status_str}")

def main():
    log_file = setup_logging()
    logging.info("=== Начало работы транспортного оптимизатора ===")

    try:
        data = load_data()
        solver, status, solve_time, x, u, v, penalty_unplaced = create_model(data)
        save_results(solver, status, solve_time, data, x, u, v, penalty_unplaced)
        logging.info("Оптимизация завершена успешно!")
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()