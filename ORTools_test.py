from ortools.linear_solver import pywraplp

import pandas as pd
import logging



def safe_float_conversion(value, default=1e6, context=""):
    """Безопасное преобразование в float с логированием ошибок"""
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        logging.warning(f"Ошибка преобразования {context} значения {value}: {str(e)}, используется {default}")
        return default


def load_and_prepare_data():
    """Загрузка и подготовка данных для второй итерации"""
    # Загрузка основных справочников
    factories = pd.read_csv('factories.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    factories.columns = ['factory']
    factories = factories['factory'].tolist()

    warehouses = pd.read_csv('warehouses.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    warehouses.columns = ['warehouse']
    warehouses = warehouses['warehouse'].tolist()

    products = pd.read_csv('products.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    products.columns = ['product']
    products = products['product'].tolist()

    weeks = pd.read_csv('week.csv', skiprows=1, header=0, delimiter=';').iloc[:, [0]]
    weeks.columns = ['week']
    weeks['week'] = weeks['week'].str.replace('W', '', regex=False)
    weeks = weeks['week'].tolist()

    # Загрузка коэффициентов с проверкой
    try:
        late_coeff_df = pd.read_csv('late_delivery_coeff.csv', skiprows=2, header=0, delimiter=';')
        late_coeff = safe_float_conversion(late_coeff_df.iloc[0, -1], 1.0, "late_delivery_coeff")
    except Exception as e:
        logging.error(f"Ошибка загрузки late_delivery_coeff: {str(e)}")
        late_coeff = 1.0

    try:
        nontarget_coeff_df = pd.read_csv('non-target_warehouse_coeff.csv', skiprows=2, header=0, delimiter=';')
        nontarget_coeff = safe_float_conversion(nontarget_coeff_df.iloc[0, -1], 1.0, "nontarget_warehouse_coeff")
    except Exception as e:
        logging.error(f"Ошибка загрузки non-target_warehouse_coeff: {str(e)}")
        nontarget_coeff = 1.0

    # Загрузка и проверка данных о стоимости
    try:
        costs_df = pd.read_csv('costs.csv', skiprows=2, header=0, delimiter=';')
        costs_df.columns = ['factory', 'warehouse', 'cost']
        costs_df['cost'] = costs_df['cost'].apply(lambda x: safe_float_conversion(x, 1e6, "costs"))
        costs_data = costs_df.set_index(['factory', 'warehouse'])['cost'].to_dict()
    except Exception as e:
        logging.error(f"Ошибка загрузки costs: {str(e)}")
        costs_data = {}

    # Загрузка данных после первой итерации
    try:
        unplaced_production_df = pd.read_csv('unplaced_production.csv', skiprows=2, header=0, delimiter=';')
        unplaced_production_df.columns = ['factory', 'product', 'week', 'unplaced']
        unplaced_production_df['week'] = unplaced_production_df['week'].str.replace('W', '', regex=False)
        production_data = unplaced_production_df.groupby(['factory', 'product', 'week'])['unplaced'].sum().to_dict()
    except Exception as e:
        logging.error(f"Ошибка загрузки unplaced_production: {str(e)}")
        production_data = {}

    try:
        demand = pd.read_csv('demand2.csv', skiprows=2, header=0, delimiter=';')
        demand.columns = ['warehouse', 'week', 'demand2']
        demand['week'] = demand['week'].str.replace('W', '', regex=False)
        demand_data = demand.groupby(['warehouse', 'week'])['demand2'].sum().to_dict()
    except Exception as e:
        logging.error(f"Ошибка загрузки demand2: {str(e)}")
        demand_data = {}

    try:
        capacity_df = pd.read_csv('capacity2.csv', skiprows=2, header=0, delimiter=';')
        capacity_df.columns = ['warehouse', 'week', 'capacity2']
        capacity_df['week'] = capacity_df['week'].str.replace('W', '', regex=False)
        capacity_df['capacity2'] = capacity_df['capacity2'].apply(lambda x: safe_float_conversion(x, 0, "capacity"))
        capacity = capacity_df.set_index(['warehouse', 'week'])['capacity2'].to_dict()
    except Exception as e:
        logging.error(f"Ошибка загрузки capacity2: {str(e)}")
        capacity = {}

    try:
        penalty_df = pd.read_csv('penalty.csv', skiprows=2, header=0, delimiter=';')
        penalty_df.columns = ['warehouse', 'penalty']
        penalty_df['penalty'] = penalty_df['penalty'].apply(lambda x: safe_float_conversion(x, 1e6, "penalty"))
        penalty_data = penalty_df.set_index('warehouse')['penalty'].to_dict()
    except Exception as e:
        logging.error(f"Ошибка загрузки penalty: {str(e)}")
        penalty_data = {}

    return {
        'factories': factories,
        'warehouses': warehouses,
        'products': products,
        'weeks': weeks,
        'late_coeff': late_coeff,
        'nontarget_coeff': nontarget_coeff,
        'costs_data': costs_data,
        'production_data': production_data,
        'demand_data': demand_data,
        'capacity': capacity,
        'penalty_data': penalty_data
    }



def create_variables(solver, data):
    """Создание переменных для модели"""
    variables = {
        'x': {},  # Основные перевозки (целевая неделя + целевой склад)
        'y': {},  # Поздние перевозки (нецелевая неделя + целевой склад)
        'z': {},  # Нецелевые перевозки (целевая неделя + нецелевой склад)
        'w': {},  # Поздние нецелевые перевозки (нецелевая неделя + нецелевой склад)
        'u': {},  # Неудовлетворенный спрос
        'v': {}  # Неразмещенная продукция
    }

    logging.info("Создание переменных для модели...")

    # Предварительная фильтрация допустимых комбинаций
    valid_combinations = []
    for i in data['factories']:
        for j in data['warehouses']:
            for p in data['products']:
                for w in data['weeks']:
                    # Проверяем, есть ли спрос или вместимость для этой пары (склад, неделя)
                    if (j, w) in data['demand_data'] or (j, w) in data['capacity']:
                        valid_combinations.append((i, j, p, w))

    logging.info(f"Создано {len(valid_combinations)} допустимых комбинаций для переменных")

    for i in data['factories']:
        for j in data['warehouses']:
            cost = data['costs_data'].get((i, j), 1e6)

            for p in data['products']:
                for w_idx, current_week in enumerate(data['weeks']):
                    # Основные перевозки (x)
                    if (j, current_week) in data['demand_data']:
                        var_name = f'x_{i}_{j}_{p}_{current_week}'
                        variables['x'][(i, j, p, current_week)] = solver.NumVar(0, solver.infinity(), var_name)

                    # Поздние перевозки (y)
                    for later_week in data['weeks'][w_idx + 1:]:
                        if (j, current_week) in data['demand_data']:
                            var_name = f'y_{i}_{j}_{p}_{current_week}_{later_week}'
                            variables['y'][(i, j, p, current_week, later_week)] = solver.NumVar(0, solver.infinity(),
                                                                                                var_name)

                    # Нецелевые перевозки (z)
                    if (j, current_week) not in data['demand_data'] and (j, current_week) in data['capacity']:
                        var_name = f'z_{i}_{j}_{p}_{current_week}'
                        variables['z'][(i, j, p, current_week)] = solver.NumVar(0, solver.infinity(), var_name)

                    # Поздние нецелевые перевозки (w)
                    for later_week in data['weeks'][w_idx + 1:]:
                        if (j, later_week) in data['capacity']:
                            var_name = f'w_{i}_{j}_{p}_{current_week}_{later_week}'
                            variables['w'][(i, j, p, current_week, later_week)] = solver.NumVar(0, solver.infinity(),
                                                                                                var_name)

    # Переменные для неудовлетворенного спроса и неразмещенной продукции
    for (j, week), dem in data['demand_data'].items():
        variables['u'][(j, week)] = solver.NumVar(0, dem, f'u_{j}_{week}')

    for (i, p, week), prod in data['production_data'].items():
        variables['v'][(i, p, week)] = solver.NumVar(0, prod, f'v_{i}_{p}_{week}')

    return variables


def setup_objective(solver, variables, data):
    """Настройка целевой функции"""
    objective = solver.Objective()

    # Основные перевозки (x)
    for (i, j, p, week), var in variables['x'].items():
        cost = data['costs_data'].get((i, j), 1e6)
        objective.SetCoefficient(var, cost)

    # Поздние перевозки (y)
    for (i, j, p, orig_week, later_week), var in variables['y'].items():
        base_cost = data['costs_data'].get((i, j), 1e6)
        weeks_diff = data['weeks'].index(later_week) - data['weeks'].index(orig_week)
        total_cost = base_cost * (1 + data['late_coeff'] * weeks_diff)
        objective.SetCoefficient(var, total_cost)

    # Нецелевые перевозки (z)
    for (i, j, p, week), var in variables['z'].items():
        base_cost = data['costs_data'].get((i, j), 1e6)
        total_cost = base_cost * (1 + data['nontarget_coeff'])
        objective.SetCoefficient(var, total_cost)

    # Поздние нецелевые перевозки (w)
    for (i, j, p, orig_week, later_week), var in variables['w'].items():
        base_cost = data['costs_data'].get((i, j), 1e6)
        weeks_diff = data['weeks'].index(later_week) - data['weeks'].index(orig_week)
        total_cost = base_cost * (1 + data['nontarget_coeff']) * (1 + data['late_coeff'] * weeks_diff)
        objective.SetCoefficient(var, total_cost)

    # Штрафы за неудовлетворенный спрос (u)
    for (j, week), var in variables['u'].items():
        penalty = data['penalty_data'].get(j, 1e6) * 1  # Большой штраф
        objective.SetCoefficient(var, penalty)

    # Штрафы за неразмещенную продукцию (v)
    max_penalty = max(data['penalty_data'].values()) * 10 if data['penalty_data'] else 1e8
    for (i, p, week), var in variables['v'].items():
        objective.SetCoefficient(var, max_penalty)

    objective.SetMinimization()



def add_constraints(solver, variables, data):
    """Добавление всех ограничений в модель с подробным логированием"""
    logging.info("Начало добавления ограничений...")

    try:
        # 1. Ограничения баланса производства
        logging.info("Добавление ограничений баланса производства...")
        for (i, p, week), prod in data['production_data'].items():
            constraint = solver.Constraint(prod, prod, f"production_{i}_{p}_{week}")

            # Основные перевозки (x)
            for (i_, j, p_, week_), var in variables['x'].items():
                if (i_, p_, week_) == (i, p, week):
                    constraint.SetCoefficient(var, 1)

            # Поздние перевозки (y)
            for (i_, j, p_, orig_week, later_week), var in variables['y'].items():
                if (i_, p_, orig_week) == (i, p, week):
                    constraint.SetCoefficient(var, 1)

            # Нецелевые перевозки (z)
            for (i_, j, p_, week_), var in variables['z'].items():
                if (i_, p_, week_) == (i, p, week):
                    constraint.SetCoefficient(var, 1)

            # Поздние нецелевые перевозки (w)
            for (i_, j, p_, orig_week, later_week), var in variables['w'].items():
                if (i_, p_, orig_week) == (i, p, week):
                    constraint.SetCoefficient(var, 1)

            # Неразмещенная продукция
            constraint.SetCoefficient(variables['v'][(i, p, week)], 1)

            logging.debug(f"Добавлено ограничение производства для {i}-{p}-W{week}")

        # 2. Ограничения вместимости складов
        logging.info("Добавление ограничений вместимости складов...")
        for (j, week), cap in data['capacity'].items():
            constraint = solver.Constraint(0, cap, f"capacity_{j}_{week}")

            # Основные перевозки (x)
            for (i_, j_, p_, week_), var in variables['x'].items():
                if j_ == j and week_ == week:
                    constraint.SetCoefficient(var, 1)

            # Нецелевые перевозки (z)
            for (i_, j_, p_, week_), var in variables['z'].items():
                if j_ == j and week_ == week:
                    constraint.SetCoefficient(var, 1)

            # Поздние перевозки (y) - если week это неделя доставки
            for (i_, j_, p_, orig_week, later_week), var in variables['y'].items():
                if j_ == j and later_week == week:
                    constraint.SetCoefficient(var, 1)

            # Поздние нецелевые перевозки (w) - если week это неделя доставки
            for (i_, j_, p_, orig_week, later_week), var in variables['w'].items():
                if j_ == j and later_week == week:
                    constraint.SetCoefficient(var, 1)

            logging.debug(f"Добавлено ограничение вместимости для {j}-W{week}")

        # 3. Ограничения удовлетворения спроса
        logging.info("Добавление ограничений спроса...")
        for (j, week), dem in data['demand_data'].items():
            constraint = solver.Constraint(dem, dem, f"demand_{j}_{week}")

            # Основные перевозки (x)
            for (i_, j_, p_, week_), var in variables['x'].items():
                if j_ == j and week_ == week:
                    constraint.SetCoefficient(var, 1)

            # Поздние перевозки (y) из предыдущих недель
            for prev_week in data['weeks'][:data['weeks'].index(week)]:
                for (i_, j_, p_, orig_week, later_week), var in variables['y'].items():
                    if j_ == j and orig_week == prev_week and later_week == week:
                        constraint.SetCoefficient(var, 1)

            # Неудовлетворенный спрос
            constraint.SetCoefficient(variables['u'][(j, week)], 1)

            logging.debug(f"Добавлено ограничение спроса для {j}-W{week}")

        logging.info("Все ограничения успешно добавлены")
        return True

    except KeyboardInterrupt:
        logging.warning("Добавление ограничений прервано пользователем!")
        return False
    except Exception as e:
        logging.error(f"Ошибка при добавлении ограничений: {str(e)}", exc_info=True)
        return False

def save_results(solver, variables, data, status):
    """Сохранение результатов решения"""
    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        logging.error("Решение не найдено или не оптимально")
        return False

    logging.info(f"Значение целевой функции: {solver.Objective().Value():.2f}")

    # Сбор результатов
    transport_plan = []

    # Основные перевозки (x)
    for (i, j, p, week), var in variables['x'].items():
        if var.solution_value() > 1e-6:
            transport_plan.append({
                'factory': i,
                'warehouse': j,
                'product': p,
                'delivery_week': f"W{week}",
                'type': 'main',
                'original_week': f"W{week}",
                'amount': var.solution_value(),
                'cost': data['costs_data'].get((i, j), 0) * var.solution_value()
            })

    # Поздние перевозки (y)
    for (i, j, p, orig_week, later_week), var in variables['y'].items():
        if var.solution_value() > 1e-6:
            weeks_diff = data['weeks'].index(later_week) - data['weeks'].index(orig_week)
            cost = data['costs_data'].get((i, j), 0) * (1 + data['late_coeff'] * weeks_diff) * var.solution_value()
            transport_plan.append({

                'factory': i,
                'warehouse': j,
                'product': p,
                'delivery_week': f"W{later_week}",
                'type': 'late',
                'original_week': f"W{orig_week}",
                'amount': var.solution_value(),
                'cost': cost
            })

    # Сохранение результатов
    try:
        pd.DataFrame(transport_plan).to_csv('second_iteration_transport_plan.csv', index=False)
        logging.info(f"Сохранен план перевозок ({len(transport_plan)} записей)")
    except Exception as e:
        logging.error(f"Ошибка сохранения результатов: {str(e)}")
        return False

    return True


def second_iteration_distribution():
    """Основная функция для второй итерации распределения"""
    logging.info("=== Начало второй итерации распределения ===")

    try:
        # 1. Загрузка и подготовка данных
        data = load_and_prepare_data()

        # 2. Создание модели
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("Не удалось создать решатель SCIP")

        # 3. Создание переменных
        variables = create_variables(solver, data)

        # 4. Настройка целевой функции
        setup_objective(solver, variables, data)

        # 5. Добавление ограничений
        add_constraints(solver, variables, data)

        # 6. Решение задачи
        logging.info("Решение модели...")
        solver.SetTimeLimit(7200 * 1000)  # 1 час
        status = solver.Solve()

        # 7. Сохранение результатов
        if not save_results(solver, variables, data, status):
            raise Exception("Не удалось сохранить результаты")

        logging.info("Вторая итерация распределения завершена успешно!")
        return True

    except Exception as e:
        logging.error(f"Ошибка во второй итерации распределения: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    second_iteration_distribution()