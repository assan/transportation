from ortools.linear_solver import pywraplp
import pandas as pd
import logging
from datetime import datetime
import os
import time


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


def load_data():
    """Загрузка всех необходимых данных"""
    logging.info("Загрузка входных данных...")

    # Загрузка справочников
    factories = pd.read_csv('factories.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    factories.columns = ['factory']

    warehouses = pd.read_csv('warehouses.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    warehouses.columns = ['warehouse']

    products = pd.read_csv('products.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    products.columns = ['product']

    weeks = pd.read_csv('week.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    weeks.columns = ['week']
    weeks['week'] = weeks['week'].str.replace('W', '', regex=False)

    # Загрузка производственных данных
    production = pd.read_csv('production.csv', skiprows=2, header=0, delimiter=';')
    production.columns = ['factory', 'product', 'week', 'amount']
    production['week'] = production['week'].str.replace('W', '', regex=False)
    production['amount'] = production['amount'].astype(float)

    demand = pd.read_csv('demand.csv', skiprows=2, header=0, delimiter=';')
    demand.columns = ['warehouse', 'week', 'demand']
    demand['week'] = demand['week'].str.replace('W', '', regex=False)
    demand['demand'] = demand['demand'].astype(float)

    capacity = pd.read_csv('capacity.csv', skiprows=2, header=0, delimiter=';')
    capacity.columns = ['warehouse', 'week', 'capacity']
    capacity['week'] = capacity['week'].str.replace('W', '', regex=False)
    capacity['capacity'] = capacity['capacity'].astype(float)

    penalty = pd.read_csv('penalty.csv', skiprows=2, header=0, delimiter=';')
    penalty.columns = ['warehouse', 'penalty']
    penalty['penalty'] = penalty['penalty'].astype(float)
    print(penalty)
    print(penalty.info())

    costs = pd.read_csv('costs.csv', skiprows=2, header=0, delimiter=';')
    costs.columns = ['factory', 'warehouse', 'cost']
    costs['cost'] = costs['cost'].astype(float)
    print(costs)
    print(costs.info())

    # Проверка данных
    check_capacity(production, capacity, weeks)

    return {
        'factories': factories['factory'].tolist(),
        'warehouses': warehouses['warehouse'].tolist(),
        'products': products['product'].tolist(),
        'weeks': weeks['week'].tolist(),
        'production': production.set_index(['factory', 'product', 'week'])['amount'].to_dict(),
        'demand': demand.set_index(['warehouse', 'week'])['demand'].to_dict(),
        'capacity': capacity.set_index(['warehouse', 'week'])['capacity'].to_dict(),
        'penalty': penalty.set_index('warehouse')['penalty'].to_dict(),
        'costs': costs.set_index(['factory', 'warehouse'])['cost'].to_dict()
    }


def check_capacity(production_df, capacity_df, weeks_df):
    """Проверка достаточности общей вместимости складов"""
    total_production = production_df.groupby('week')['amount'].sum()
    total_capacity = capacity_df.groupby('week')['capacity'].sum()

    for week in weeks_df['week']:
        prod = total_production.get(week, 0)
        cap = total_capacity.get(week, 0)
        if prod > cap:
            logging.warning(f"Неделя {week}: Производство {prod} > Вместимость {cap} (дефицит {prod - cap})")


def create_model(data):
    """Создание и решение оптимизационной модели"""
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Не удалось создать решатель SCIP")

    # Создание переменных
    x = {}  # Перевозки
    u = {}  # Неудовлетворенный спрос
    v = {}  # Неразмещенная продукция

    logging.info("Создание переменных...")
    for i in data['factories']:
        for j in data['warehouses']:
            for p in data['products']:
                for w in data['weeks']:
                    x[(i, j, p, w)] = solver.NumVar(0, solver.infinity(), f'x_{i}_{j}_{p}_{w}')

    for j in data['warehouses']:
        for w in data['weeks']:
            if (j, w) in data['demand']:
                u[(j, w)] = solver.NumVar(0, data['demand'][(j, w)], f'u_{j}_{w}')

    for i in data['factories']:
        for p in data['products']:
            for w in data['weeks']:
                if (i, p, w) in data['production']:
                    v[(i, p, w)] = solver.NumVar(0, data['production'][(i, p, w)], f'v_{i}_{p}_{w}')

    # Целевая функция
    logging.info("Формирование целевой функции...")
    objective = solver.Objective()

    # Транспортные затраты
    for (i, j), cost in data['costs'].items():
        for p in data['products']:
            for w in data['weeks']:
                if (i, j, p, w) in x:
                    objective.SetCoefficient(x[(i, j, p, w)], cost)

    # Штрафы за неудовлетворенный спрос
    for (j, w), var in u.items():
        if j in data['penalty']:
            objective.SetCoefficient(var, data['penalty'][j])

    # Штраф за неразмещенную продукцию (значительно выше штрафа за неудовлетворенный спрос)
    penalty_unplaced = max(data['penalty'].values()) * 1000 if data['penalty'] else 10000
    for (i, p, w), var in v.items():
        objective.SetCoefficient(var, penalty_unplaced)

    objective.SetMinimization()

    # Ограничения
    logging.info("Добавление ограничений...")

    # 1. Баланс производства (часть может остаться неразмещенной)
    for i in data['factories']:
        for p in data['products']:
            for w in data['weeks']:
                if (i, p, w) in data['production']:
                    constraint = solver.Constraint(
                        data['production'][(i, p, w)], data['production'][(i, p, w)],
                        f"Production_{i}_{p}_{w}")

                    for j in data['warehouses']:
                        if (i, j, p, w) in x:
                            constraint.SetCoefficient(x[(i, j, p, w)], 1)

                    constraint.SetCoefficient(v[(i, p, w)], 1)

    # 2. Вместимость складов
    for j in data['warehouses']:
        for w in data['weeks']:
            if (j, w) in data['capacity']:
                constraint = solver.Constraint(
                    0, data['capacity'][(j, w)],
                    f"Capacity_{j}_{w}")

                for i in data['factories']:
                    for p in data['products']:
                        if (i, j, p, w) in x:
                            constraint.SetCoefficient(x[(i, j, p, w)], 1)

    # 3. Удовлетворение спроса (может быть частичным)
    for j in data['warehouses']:
        for w in data['weeks']:
            if (j, w) in data['demand']:
                constraint = solver.Constraint(
                    data['demand'][(j, w)], data['demand'][(j, w)],
                    f"Demand_{j}_{w}")

                for i in data['factories']:
                    for p in data['products']:
                        if (i, j, p, w) in x:
                            constraint.SetCoefficient(x[(i, j, p, w)], 1)

                constraint.SetCoefficient(u[(j, w)], 1)

    # Решение задачи
    logging.info("Решение модели...")
    solver.SetTimeLimit(7200 * 1000)  # 2 часа
    solver.EnableOutput()

    start_time = time.time()
    status = solver.Solve()
    solve_time = time.time() - start_time

    return solver, status, solve_time, x, u, v, penalty_unplaced


def save_results(solver, status, solve_time, data, x, u, v, penalty_unplaced):
    """Сохранение и анализ результатов"""
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
        logging.info(f"Значение целевой функции: {solver.Objective().Value():.2f}")

        # Сохранение плана перевозок
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

        # Анализ неудовлетворенного спроса
        unsatisfied = []
        for (j, w), var in u.items():
            if var.solution_value() > 1e-6:
                unsatisfied.append({
                    'warehouse': j,
                    'week': f"W{w}",
                    'demand': data['demand'][(j, w)],
                    'unsatisfied': var.solution_value(),
                    'penalty': var.solution_value() * data['penalty'][j]
                })

        if unsatisfied:
            pd.DataFrame(unsatisfied).to_csv('unsatisfied_demand_ortools.csv', index=False)
            total_unsatisfied = sum(item['unsatisfied'] for item in unsatisfied)
            logging.warning(f"Общий неудовлетворенный спрос: {total_unsatisfied:.2f}")

        # Анализ неразмещенной продукции
        unplaced = []
        for (i, p, w), var in v.items():
            if var.solution_value() > 1e-6:
                unplaced.append({
                    'factory': i,
                    'product': p,
                    'week': f"W{w}",
                    'production': data['production'][(i, p, w)],
                    'unplaced': var.solution_value(),
                    'penalty': var.solution_value() * penalty_unplaced
                })

        if unplaced:
            pd.DataFrame(unplaced).to_csv('unplaced_production_ortools.csv', index=False)
            total_unplaced = sum(item['unplaced'] for item in unplaced)
            logging.error(f"Общий неразмещенный объем: {total_unplaced:.2f}")

        # Сводная статистика
        logging.info("\n=== Сводная статистика ===")
        logging.info(f"Всего произведено: {sum(data['production'].values()):.2f}")
        logging.info(f"Всего размещено: {sum(item['amount'] for item in transport_plan):.2f}")
        if unsatisfied:
            logging.info(
                f"Процент удовлетворенного спроса: {(1 - total_unsatisfied / sum(data['demand'].values())) * 100:.2f}%")
        if unplaced:
            logging.info(
                f"Процент размещенной продукции: {(1 - total_unplaced / sum(data['production'].values())) * 100:.2f}%")

    else:
        raise Exception(f"Не удалось найти решение. Статус: {status_str}")


def main():
    """Основная функция"""
    log_file = setup_logging()
    logging.info("=== Начало работы транспортного оптимизатора ===")

    try:
        # Загрузка данных
        data = load_data()

        # Создание и решение модели
        solver, status, solve_time, x, u, v, penalty_unplaced = create_model(data)

        # Сохранение результатов
        save_results(solver, status, solve_time, data, x, u, v, penalty_unplaced)

        logging.info("Оптимизация завершена успешно!")
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()