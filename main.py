import csv
import random
import time
import numpy as np
from typing import List, Tuple, Optional
import statistics

def write_list_to_csv(data: List[int], filename: str) -> None:
    """
    Записывает список чисел в CSV файл.
    
    Args:
        data: Список чисел для записи
        filename: Имя файла для записи
    """
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['number'])
        for item in data:
            writer.writerow([item])

def generate_random_numbers_list(n: int, a: int, b: int, even_only: bool = True) -> List[int]:
    """
    Генерирует список случайных чисел без повторов используя list и random.randint.
    
    Args:
        n: Количество чисел для генерации
        a: Нижняя граница интервала
        b: Верхняя граница интервала
        even_only: True для четных чисел, False для нечетных
    
    Returns:
        Список уникальных случайных чисел
    """
    numbers = []
    attempts = 0
    max_attempts = n * 100  # Защита от бесконечного цикла
    
    while len(numbers) < n and attempts < max_attempts:
        num = random.randint(a, b)
        attempts += 1
        if even_only and num % 2 == 0 and num not in numbers:
            numbers.append(num)
        elif not even_only and num % 2 != 0 and num not in numbers:
            numbers.append(num)
    
    if len(numbers) < n:
        raise RuntimeError(f"Не удалось сгенерировать {n} чисел за {max_attempts} попыток")
    
    return numbers

def generate_random_numbers_set(n: int, a: int, b: int, even_only: bool = True) -> List[int]:
    """
    Генерирует список случайных чисел без повторов используя set.
    
    Args:
        n: Количество чисел для генерации
        a: Нижняя граница интервала
        b: Верхняя граница интервала
        even_only: True для четных чисел, False для нечетных
    
    Returns:
        Список уникальных случайных чисел
    """
    numbers = set()
    attempts = 0
    max_attempts = n * 100  
    
    while len(numbers) < n and attempts < max_attempts:
        num = random.randint(a, b)
        attempts += 1
        if even_only and num % 2 == 0:
            numbers.add(num)
        elif not even_only and num % 2 != 0:
            numbers.add(num)
    
    if len(numbers) < n:
        raise RuntimeError(f"Не удалось сгенерировать {n} чисел за {max_attempts} попыток")
    
    return list(numbers)

def generate_random_numbers_numpy(n: int, a: int, b: int, even_only: bool = True) -> List[int]:
    """
    Генерирует список случайных чисел без повторов используя numpy без циклов.
    
    Args:
        n: Количество чисел для генерации
        a: Нижняя граница интервала
        b: Верхняя граница интервала
        even_only: True для четных чисел, False для нечетных
    
    Returns:
        Список уникальных случайных чисел
    """
    
    if even_only:
        
        start = a if a % 2 == 0 else a + 1
        all_numbers = np.arange(start, b + 1, 2)
    else:
        
        start = a if a % 2 != 0 else a + 1
        all_numbers = np.arange(start, b + 1, 2)
    
    
    if len(all_numbers) < n:
        raise ValueError(f"Недостаточно {'четных' if even_only else 'нечетных'} чисел в диапазоне")
    
    
    selected = np.random.choice(all_numbers, size=n, replace=False)
    return selected.tolist()

def measure_execution_time_precise(func, *args, runs: int = 10) -> Tuple[List[int], float]:
    """
    Точно измеряет время выполнения функции с несколькими запусками.
    
    Args:
        func: Функция для измерения
        *args: Аргументы функции
        runs: Количество запусков для усреднения
    
    Returns:
        Кортеж (результат последнего запуска, среднее время выполнения в секундах)
    """
    times = []
    result = []  
    
    for _ in range(runs):
        start_time = time.perf_counter()
        try:
            result = func(*args)
        except Exception:
            # Если функция не выполнилась, повторно поднимаем исключение
            raise
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Убираем выбросы и берем медиану для более стабильного результата
    if len(times) >= 3:
        times.sort()
        times = times[1:-1]  # Убираем минимум и максимум
    
    avg_time = statistics.mean(times) if times else 0.0
    return result, avg_time

def run_comparison_test(n: int, a: int, b: int, even_only: bool = True) -> Tuple[float, float, float]:
    """
    Запускает тест сравнения производительности всех трех методов.
    
    Args:
        n: Количество чисел для генерации
        a: Нижняя граница интервала
        b: Верхняя граница интервала
        even_only: True для четных чисел, False для нечетных
    
    Returns:
        Кортеж времен выполнения (time_list, time_set, time_numpy)
    """
    print(f"\nТест для n={n}, диапазон ({a}, {b}), {'четные' if even_only else 'нечетные'} числа:")
    
    # Определяем количество прогонов в зависимости от n
    runs = max(3, min(100, 1000 // max(1, n // 10)))
    print(f"Количество прогонов для усреднения: {runs}")
    
    # Тест с list
    try:
        result_list, time_list = measure_execution_time_precise(generate_random_numbers_list, n, a, b, even_only, runs=runs)
        print(f"List метод: {time_list:.6f} секунд (среднее)")
        write_list_to_csv(result_list, f"random_numbers_list_{n}.csv")
    except Exception as e:
        print(f"List метод: Ошибка - {e}")
        time_list = float('inf')
    
    # Тест с set
    try:
        result_set, time_set = measure_execution_time_precise(generate_random_numbers_set, n, a, b, even_only, runs=runs)
        print(f"Set метод: {time_set:.6f} секунд (среднее)")
        write_list_to_csv(result_set, f"random_numbers_set_{n}.csv")
    except Exception as e:
        print(f"Set метод: Ошибка - {e}")
        time_set = float('inf')
    
    # Тест с numpy
    try:
        result_numpy, time_numpy = measure_execution_time_precise(generate_random_numbers_numpy, n, a, b, even_only, runs=runs)
        print(f"NumPy метод: {time_numpy:.6f} секунд (среднее)")
        write_list_to_csv(result_numpy, f"random_numbers_numpy_{n}.csv")
    except Exception as e:
        print(f"NumPy метод: Ошибка - {e}")
        time_numpy = float('inf')
    
    return time_list, time_set, time_numpy

def write_performance_report(results: List[Tuple[int, Tuple[float, float, float]]], even_only: bool) -> None:
    """
    Записывает отчет о производительности в markdown файл.
    
    Args:
        results: Список результатов тестов
        even_only: Флаг четных/нечетных чисел
    """
    with open("performance_report.md", "w", encoding="utf-8") as f:
        f.write(" Отчет о производительности генерации случайных чисел\n\n")
        f.write(f"Тип чисел: {'четные' if even_only else 'нечетные'}\n\n")
        
        f.write("Результаты тестирования\n\n")
        f.write("| n | List (мкс) | Set (мкс) | NumPy (мкс) | Быстрейший |\n")
        f.write("|---|------------|-----------|-------------|-------------|\n")
        
        for n, (time_list, time_set, time_numpy) in results:
            # Переводим в микросекунды для лучшей читаемости
            list_micro = time_list * 1_000_000 if time_list != float('inf') else float('inf')
            set_micro = time_set * 1_000_000 if time_set != float('inf') else float('inf')
            numpy_micro = time_numpy * 1_000_000 if time_numpy != float('inf') else float('inf')
            
            # Определяем быстрейший метод
            times_dict = {'List': time_list, 'Set': time_set, 'NumPy': time_numpy}
            valid_times = {k: v for k, v in times_dict.items() if v != float('inf')}
            fastest = min(valid_times.keys(), key=lambda k: valid_times[k]) if valid_times else "N/A"
            
            list_str = f"{list_micro:.1f}" if list_micro != float('inf') else "∞"
            set_str = f"{set_micro:.1f}" if set_micro != float('inf') else "∞"
            numpy_str = f"{numpy_micro:.1f}" if numpy_micro != float('inf') else "∞"
            
            f.write(f"| {n} | {list_str} | {set_str} | {numpy_str} | **{fastest}** |\n")
        
        f.write("\n Анализ результатов\n\n")
        
       
        if results:
            f.write(" Фактическая производительность:\n\n")
            for n, (time_list, time_set, time_numpy) in results:
                times = {'List': time_list, 'Set': time_set, 'NumPy': time_numpy}
                valid_times = {k: v for k, v in times.items() if v != float('inf')}
                if valid_times:
                    fastest = min(valid_times.keys(), key=lambda k: valid_times[k])
                    slowest = max(valid_times.keys(), key=lambda k: valid_times[k])
                    
                    f.write(f"**n={n}**: Быстрейший - {fastest}, медленнейший - {slowest}\n")
                    
                    if len(valid_times) >= 2:
                        fastest_time = valid_times[fastest]
                        slowest_time = valid_times[slowest]
                        if fastest_time > 0:
                            ratio = slowest_time / fastest_time
                            f.write(f"  - {slowest} медленнее {fastest} в {ratio:.1f} раз\n")
                f.write("\n")
        
        f.write(" Теоретический анализ vs Реальность:\n\n")
        f.write("1. Для малых n (< 1000):\n")
        f.write("   - NumPy может быть медленнее из-за накладных расходов на инициализацию\n")
        f.write("   - List и Set показывают похожую производительность\n")
        f.write("   - Разница в производительности может быть незначительной\n\n")
        
        f.write("2. Для средних n (1000-10000):\n")
        f.write("   - Начинает проявляться O(n²) сложность List метода\n")
        f.write("   - Set метод становится заметно быстрее List\n")
        f.write("   - NumPy может показать преимущество\n\n")
        
        f.write("3. Для больших n (> 10000):\n")
        f.write("   - List метод становится неприемлемо медленным\n")
        f.write("   - Set и NumPy методы значительно опережают List\n")
        f.write("   - NumPy обычно показывает лучшую производительность\n\n")
        
        f.write("Рекомендации:\n\n")
        f.write("- n < 100: Используйте любой метод\n")
        f.write("- 100 ≤ n < 10000: Предпочтительнее Set метод\n")
        f.write("- n ≥ 10000: Используйте NumPy метод\n")

def main() -> None:
    """
    Основная функция программы.
    """
    print("Генерация случайных чисел без повторов")
    
    # Получаем параметры от пользователя
    try:
        n = int(input("Введите количество чисел: "))
        a = int(input("Введите нижнюю границу интервала: "))
        b = int(input("Введите верхнюю границу интервала: "))
        variant = input("Введите вариант (четный/нечетный): ").lower()
        even_only = variant.startswith('ч')
        
        print(f"\nПараметры: n={n}, интервал ({a}, {b}), {'четные' if even_only else 'нечетные'} числа")
        
        # Проверяем корректность входных данных
        if even_only:
            first_even = a if a % 2 == 0 else a + 1
            last_even = b if b % 2 == 0 else b - 1
            if first_even <= last_even:
                available_numbers = (last_even - first_even) // 2 + 1
            else:
                available_numbers = 0
        else:
            first_odd = a if a % 2 != 0 else a + 1
            last_odd = b if b % 2 != 0 else b - 1
            if first_odd <= last_odd:
                available_numbers = (last_odd - first_odd) // 2 + 1
            else:
                available_numbers = 0
        
        print(f"В диапазоне [{a}, {b}] доступно {available_numbers} {'четных' if even_only else 'нечетных'} чисел")
        
        if n > available_numbers:
            print(f"\n❌ Ошибка: недостаточно чисел в диапазоне")
            print("💡 Попробуйте: n=1000, a=1, b=10000, нечетный")
            return
        
        # Запускаем тесты
        results: List[Tuple[int, Tuple[float, float, float]]] = []
        
        # Основной тест
        time_results = run_comparison_test(n, a, b, even_only)
        results.append((n, time_results))
        
        # Дополнительные тесты для демонстрации разницы
        if available_numbers >= 1000:
            additional_tests = [100, 500, 1000, 5000]
            for test_n in additional_tests:
                if test_n <= available_numbers and test_n != n:
                    time_results = run_comparison_test(test_n, a, b, even_only)
                    results.append((test_n, time_results))
        
        # Записываем результаты
        write_performance_report(results, even_only)
        print(f"\n📊 Результаты сохранены в файл 'performance_report.md'")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()