import pytest
import numpy as np
from main import generate_random_numbers_list, generate_random_numbers_set, generate_random_numbers_numpy

def test_list_generation():
    """Тест генерации с использованием list"""
    result = generate_random_numbers_list(100, 1, 1000, True)
    assert len(result) == 100
    assert all(x % 2 == 0 for x in result)
    assert len(set(result)) == 100  # Все элементы уникальны

def test_set_generation():
    """Тест генерации с использованием set"""
    result = generate_random_numbers_set(100, 1, 1000, False)
    assert len(result) == 100
    assert all(x % 2 == 1 for x in result)
    assert len(set(result)) == 100

def test_numpy_generation():
    """Тест генерации с использованием numpy"""
    result = generate_random_numbers_numpy(100, 1, 1000, True)
    assert len(result) == 100
    assert all(x % 2 == 0 for x in result)
    assert len(set(result)) == 100

def test_edge_cases():
    """Тест граничных случаев"""
    # Проверка малого количества чисел
    result = generate_random_numbers_numpy(1, 2, 2, True)
    assert result == [2]
    
    # Проверка ошибки при недостатке чисел
    with pytest.raises(ValueError):
        generate_random_numbers_numpy(100, 1, 10, True)