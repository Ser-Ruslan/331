import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Union, Any
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


ScalarOrArray = Union[float, NDArray[Any]]

def f(x: ScalarOrArray, y: ScalarOrArray) -> ScalarOrArray:
    """
    Целевая функция для минимизации.

    Аргументы:
        x (float или ndarray): координата X (может быть numpy-массивом для построения контуров).
        y (float или ndarray): координата Y (может быть numpy-массивом для построения контуров).

    Возвращает:
        float или ndarray: значение функции в точке/точках (x, y).
    """
    # Используем операции numpy — они корректно работают и для скаляров, и для массивов
    return (x**4) / 4 + x**3 - (13 * x**2) / 2 - 15 * x + \
           (y**4) / 4 - (7 * y**3) / 3 + (7 * y**2) / 2 + 15 * y


def grad_dx(x: float, y: float) -> float:
    """
    Частная производная функции f по x (для одномерных значений).
    """
    return x**3 + 3 * x**2 - 13 * x - 15


def grad_dy(x: float, y: float) -> float:
    """
    Частная производная функции f по y (для одномерных значений).
    """
    return y**3 - 7 * y**2 + 7 * y + 15


def gradient_descent(x0: float, y0: float,
                     learning_rate: float = 0.01,
                     steps: int = 100) -> List[Tuple[float, float]]:
    """
    Выполняет метод градиентного спуска для поиска минимума функции f.

    Аргументы:
        x0 (float): начальная координата X.
        y0 (float): начальная координата Y.
        learning_rate (float): скорость обучения (длина шага).
        steps (int): максимальное количество итераций.

    Возвращает:
        список кортежей (x, y) — траектория.
    """
    trajectory: List[Tuple[float, float]] = [(x0, y0)]
    x, y = x0, y0
    for _ in range(steps):
        gx = grad_dx(x, y)
        gy = grad_dy(x, y)
        x = x - learning_rate * gx
        y = y - learning_rate * gy
        trajectory.append((x, y))
    return trajectory


def draw_trajectory(trajectory: List[Tuple[float, float]],
                    xlim: Tuple[float, float] = (-6.0, 6.0),
                    ylim: Tuple[float, float] = (-6.0, 6.0),
                    grid_points: int = 400) -> Figure:
    """
    Строит контурный график функции и траекторию градиентного спуска.

    Аргументы:
        trajectory: список точек траектории.
        xlim, ylim: границы по оси x и y.
        grid_points: число точек по каждой оси для сетки.

    Возвращает:
        matplotlib.figure.Figure с построенным графиком.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_points)
    ys = np.linspace(ylim[0], ylim[1], grid_points)
    X, Y = np.meshgrid(xs, ys)

    
    Z = f(X, Y)

    fig: Figure
    fig, ax = plt.subplots(figsize=(6, 5))
    CS = ax.contour(X, Y, Z, levels=50)
    ax.clabel(CS, inline=1, fontsize=8)

    tx, ty = zip(*trajectory)
    ax.plot(tx, ty, marker='o', linewidth=1, markersize=4)
    ax.set_title("Градиентный спуск")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)
    return fig


class GradientDescentApp:
    """
    Графическое приложение для визуализации градиентного спуска.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Инициализация GUI.

        Аргументы:
            root: главное окно tkinter.
        """
        self.root = root
        root.title("Метод градиентного спуска")

        
        ttk.Label(root, text="Начальная точка X:").grid(row=0, column=0, sticky='w', padx=4, pady=2)
        ttk.Label(root, text="Начальная точка Y:").grid(row=1, column=0, sticky='w', padx=4, pady=2)
        ttk.Label(root, text="Скорость обучения:").grid(row=2, column=0, sticky='w', padx=4, pady=2)
        ttk.Label(root, text="Количество шагов:").grid(row=3, column=0, sticky='w', padx=4, pady=2)

        
        self.entry_x = ttk.Entry(root, width=20)
        self.entry_y = ttk.Entry(root, width=20)
        self.entry_lr = ttk.Entry(root, width=20)
        self.entry_steps = ttk.Entry(root, width=20)

        
        self.entry_x.insert(0, "-4")
        self.entry_y.insert(0, "-4")
        self.entry_lr.insert(0, "0.01")
        self.entry_steps.insert(0, "100")

        
        self.entry_x.grid(row=0, column=1, padx=4, pady=2)
        self.entry_y.grid(row=1, column=1, padx=4, pady=2)
        self.entry_lr.grid(row=2, column=1, padx=4, pady=2)
        self.entry_steps.grid(row=3, column=1, padx=4, pady=2)

        
        self.button_run = ttk.Button(root, text="Запустить", command=self.run)
        self.button_run.grid(row=4, column=0, columnspan=2, pady=8)

        self.canvas: Union[FigureCanvasTkAgg, None] = None

    def run(self) -> None:
        """
        Считать параметры, выполнить градиентный спуск и показать график.
        """
        try:
            x0 = float(self.entry_x.get())
            y0 = float(self.entry_y.get())
            lr = float(self.entry_lr.get())
            steps = int(self.entry_steps.get())
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Параметры должны быть числами")
            return

        trajectory = gradient_descent(x0, y0, learning_rate=lr, steps=steps)
        fig = draw_trajectory(trajectory)

        
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2, padx=4, pady=4)


if __name__ == "__main__":
    root = tk.Tk()
    app = GradientDescentApp(root)
    root.mainloop()
