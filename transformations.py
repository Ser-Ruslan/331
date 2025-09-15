import numpy as np
from typing import Tuple

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Вычисляет сигмоиду для каждого элемента массива."""
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """Вычисляет ReLU для каждого элемента массива."""
    return np.maximum(0, x)

def tanh(x: np.ndarray) -> np.ndarray:
    """Вычисляет гиперболический тангенс для каждого элемента массива."""
    return np.tanh(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Вычисляет softmax для массива."""
    exp_x = np.exp(x - np.max(x))  # Для численной стабильности
    return exp_x / exp_x.sum()

def normalize(x: np.ndarray) -> np.ndarray:
    """Нормализует значения массива к диапазону [0, 1]."""
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)

def standardize(x: np.ndarray) -> np.ndarray:
    """Стандартизирует массив (z-score)."""
    return (x - np.mean(x)) / np.std(x)

def softplus(x: np.ndarray) -> np.ndarray:
    """Вычисляет softplus для каждого элемента массива."""
    return np.log(1 + np.exp(x))

def gaussian(x: np.ndarray) -> np.ndarray:
    """Вычисляет гауссову функцию для каждого элемента массива."""
    return np.exp(-x**2)

def safe_product_calculation(arr: np.ndarray) -> Tuple[float, float]:
    """
    Безопасно вычисляет сумму и произведение с обработкой больших чисел.
    
    Returns:
        Кортеж (сумма, произведение) или (сумма, логарифм произведения)
    """
    try:
        # Вычисляем сумму
        sum_val = float(np.sum(arr))
        
        # Пытаемся вычислить произведение
        product = np.prod(arr)
        if np.isfinite(product) and abs(product) > 1e-300:  # Проверяем на переполнение и слишком малые значения
            return sum_val, float(product)
        else:
            # Используем логарифмы для избежания переполнения
            log_product = np.sum(np.log(np.abs(arr) + 1e-12))
            return sum_val, float(log_product)
    except:
        # В случае ошибки используем логарифмический подход
        log_product = np.sum(np.log(np.abs(arr) + 1e-12))
        return sum_val, float(log_product)

def process_array(arr: np.ndarray, transformation: str) -> Tuple[float, float]:
    """
    Применяет преобразование к массиву и возвращает сумму и произведение.
    
    Args:
        arr: Входной массив
        transformation: Тип преобразования
        
    Returns:
        Кортеж (сумма, произведение)
    """
    transformations = {
        'sigmoid': sigmoid,
        'relu': relu,
        'tanh': tanh,
        'softmax': softmax,
        'normalize': normalize,
        'standardize': standardize,
        'softplus': softplus,
        'gaussian': gaussian
    }
    
    if transformation not in transformations:
        raise ValueError(f"Неизвестное преобразование: {transformation}")
    
    transformed = transformations[transformation](arr)
    return safe_product_calculation(transformed)