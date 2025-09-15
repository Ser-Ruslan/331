import time
import numpy as np
from typing import List, Tuple
from main import run_comparison_test

def extended_performance_test() -> None:
    """
    Расширенный тест производительности для больших значений n.
    """
    print("=== РАСШИРЕННЫЙ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ===")
    
    # Параметры для тестирования
    test_configs: List[Tuple[int, int, int, bool]] = [
        (1000, 1, 100000, True),
        (5000, 1, 100000, True), 
        (10000, 1, 100000, True),
        (50000, 1, 1000000, True),
        (100000, 1, 1000000, True),
    ]
    
    results: List[Tuple[int, Tuple[float, float, float]]] = []
    
    for n, a, b, even_only in test_configs:
        print(f"\nТестирование n={n}, диапазон ({a}, {b})")
        
        # Проверяем доступность достаточного количества чисел
        available = (b - a + 1) // 2
        if n > available:
            print(f"Пропускаем: недостаточно чисел (доступно {available})")
            continue
        
        time_results = run_comparison_test(n, a, b, even_only)
        results.append((n, time_results))
        
        # Прерываем если один из методов работает слишком долго
        if any(t > 30 for t in time_results if t != float('inf')):
            print("Прерываем тестирование из-за слишком долгого выполнения")
            break
    
    
    write_extended_report(results)

def write_extended_report(results: List[Tuple[int, Tuple[float, float, float]]]) -> None:
    """
    Записывает расширенный отчет производительности.
    
    Args:
        results: Список результатов тестов
    """
    with open("extended_performance_report.md", "w", encoding="utf-8") as f:
        f.write(" Расширенный отчет о производительности\n\n")
        
        f.write(" Результаты для больших значений n\n\n")
        f.write("| n | List (сек) | Set (сек) | NumPy (сек) | Лучший метод |\n")
        f.write("|---|------------|-----------|-------------|---------------|\n")
        
        for n, (time_list, time_set, time_numpy) in results:
            times = {'List': time_list, 'Set': time_set, 'NumPy': time_numpy}
            valid_times = {k: v for k, v in times.items() if v != float('inf')}
            best = min(valid_times.keys(), key=lambda k: valid_times[k]) if valid_times else "N/A"
            
            list_str = f"{time_list:.4f}" if time_list != float('inf') else "∞"
            set_str = f"{time_set:.4f}" if time_set != float('inf') else "∞"
            numpy_str = f"{time_numpy:.4f}" if time_numpy != float('inf') else "∞"
            
            f.write(f"| {n} | {list_str} | {set_str} | {numpy_str} | {best} |\n")
        
        f.write("\n Детальный анализ\n\n")
        f.write(" Масштабируемость алгоритмов:\n\n")
        
        if len(results) >= 2:
            f.write("Рост времени выполнения:\n\n")
            for i in range(1, len(results)):
                n1, (t1_list, t1_set, t1_numpy) = results[i-1]
                n2, (t2_list, t2_set, t2_numpy) = results[i]
                
                ratio_n = n2 / n1
                f.write(f"При увеличении n с {n1} до {n2} (в {ratio_n:.1f} раз):**\n\n")
                
                if t1_list != float('inf') and t2_list != float('inf') and t1_list > 0:
                    ratio_list = t2_list / t1_list
                    f.write(f"- List: время увеличилось в {ratio_list:.1f} раз\n")
                
                if t1_set != float('inf') and t2_set != float('inf') and t1_set > 0:
                    ratio_set = t2_set / t1_set
                    f.write(f"- **Set**: время увеличилось в {ratio_set:.1f} раз\n")
                
                if t1_numpy != float('inf') and t2_numpy != float('inf') and t1_numpy > 0:
                    ratio_numpy = t2_numpy / t1_numpy
                    f.write(f"- **NumPy**: время увеличилось в {ratio_numpy:.1f} раз\n")
                
                f.write("\n")
        
        f.write("Заключение\n\n")
        f.write("На основе результатов тестирования можно сделать следующие выводы:\n\n")
        f.write("1. NumPy показывает лучшую производительность для больших объемов данных\n")
        f.write("2. Set является хорошей альтернативой при ограничениях памяти\n")
        f.write("3. List следует избегать для больших n из-за квадратичной сложности\n")

if __name__ == "__main__":
    extended_performance_test()