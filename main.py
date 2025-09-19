# pip install numpy pandas matplotlib xlsxwriter

import threading
import queue
import os
import math
import argparse
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f_variant1(x: np.ndarray) -> float:
    """Целевая функция варианта 1."""
    x1, x2, x3 = x
    val = 0.4 * (x1 - 0.9) ** 4 + 0.4 * (x2 - 0.6) ** 4 + 0.6 * (x3 - 0.9) ** 4 + 4.0
    return math.log(val)

def simulated_annealing(
    f,
    x0: np.ndarray,
    T0: float = 100.0,
    Tmin: float = 1e-4,
    alpha: float = 0.95,
    attempts_per_T: int = 200,
    step_scale: float = 0.5,
    max_iter: int = 20000,
    rng: Optional[np.random.Generator] = None,
    progress_callback=None,
    log_callback=None,
    log_interval: int = 500,
    progress_interval: int = 100,    # <-- новый параметр: как часто вызывать прогресс_callback
) -> Dict[str, Any]:
    """Имитация отжига с возможностью логирования и обновления прогресса.

    progress_callback(iteration, max_iter) вызывается не чаще, чем progress_interval.
    log_callback вызывается как и раньше каждые log_interval итераций.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_curr = np.array(x0, dtype=float)
    f_curr = f(x_curr)
    T = float(T0)

    trace = []
    iter_count = 0

    if not max_iter or max_iter <= 0:
        max_iter = 10**9

    # Защита от нуля/отрицательного шага interval
    if progress_interval is None or progress_interval <= 0:
        progress_interval = max(1, max_iter // 1000)

    # Локальные переменные для ускорения циклов
    local_f = f
    local_rng = rng
    local_exp = math.exp
    local_append = trace.append
    local_shape = x_curr.shape

    while T > Tmin and iter_count < max_iter:
        for _ in range(attempts_per_T):
            # Предлагаем новый шаг (нормальное распределение)
            step = local_rng.normal(scale=step_scale, size=local_shape)
            x_new = x_curr + step
            f_new = local_f(x_new)
            df = f_new - f_curr
            accept = False
            if df <= 0:
                accept = True
            else:
                p = local_exp(-df / T)
                if local_rng.random() < p:
                    accept = True
            if accept:
                x_curr = x_new
                f_curr = f_new
            local_append((iter_count, T, *x_curr.tolist(), f_curr, df))

            iter_count += 1

            # Прогресс обновляем реже (это главное ускорение)
            if progress_callback is not None and (iter_count % progress_interval == 0 or iter_count == 1 or iter_count >= max_iter):
                try:
                    progress_callback(iter_count, max_iter)
                except Exception:
                    pass

            # Логируем реже — как задаётся через log_interval
            if log_callback is not None and (iter_count % log_interval == 0):
                try:
                    log_callback(f"Итерация {iter_count}, T={T:.6g}, f={f_curr:.6g}")
                except Exception:
                    pass

            if iter_count >= max_iter:
                break
        T *= alpha

    df_trace = pd.DataFrame(trace, columns=["итерация", "температура", "x1", "x2", "x3", "f", "дельта_f"])
    result = {"x": x_curr, "f": f_curr, "trace": df_trace}
    return result

def run_multiple_runs(
    n_runs: int,
    seed: int,
    param_kwargs: Dict[str, Any],
    init_bounds: tuple = (-1.5, 2.0),
    progress_callback=None,
    log_callback=None
):
    """Несколько независимых запусков с логом и прогресс-колбэками."""
    rng_master = np.random.default_rng(seed)
    all_results = []

    # ожидаемое общее число итераций
    per_run_max = param_kwargs.get("max_iter", 20000)
    total_estimated = n_runs * per_run_max

    # Заберём (и удалим) progress_interval/log_interval из param_kwargs, если они там были,
    # чтобы не передавать их дважды при вызове simulated_annealing
    progress_interval = param_kwargs.pop("progress_interval", None)
    log_interval = param_kwargs.pop("log_interval", None)

    for run in range(1, n_runs + 1):
        x0 = rng_master.uniform(init_bounds[0], init_bounds[1], size=3)
        rng = np.random.default_rng(seed + run)

        # локальная обёртка: переводим локальный прогресс (iteration) в глобальный
        def local_progress(iteration, local_max):
            # compute global iteration count: (run-1)*local_max + iteration
            global_iter = (run - 1) * local_max + iteration
            if progress_callback:
                try:
                    progress_callback(global_iter, total_estimated)
                except Exception:
                    pass

        def local_log(s):
            if log_callback:
                log_callback(f"[Прогон {run}] {s}")

        # Если progress_interval не был задан явно, вычислим разумный дефолт
        if progress_interval is None:
            local_progress_interval = max(1, per_run_max // 1000)
        else:
            local_progress_interval = progress_interval

        if log_interval is None:
            local_log_interval = max(1, per_run_max // 200)
        else:
            local_log_interval = log_interval

        res = simulated_annealing(
            f_variant1,
            x0,
            rng=rng,
            progress_callback=local_progress,
            log_callback=local_log,
            progress_interval=local_progress_interval,
            log_interval=local_log_interval,
            **param_kwargs
        )
        res_summary = {"run": run, "x": res["x"], "f": res["f"], "trace": res["trace"]}
        all_results.append(res_summary)
        if log_callback:
            log_callback(f"Завершён прогон {run}: f={res['f']:.6g}")

    best = min(all_results, key=lambda r: float(r["f"]))
    return all_results, best


def save_results_to_excel(out_dir: str, all_results, best_result, params: dict):
    """Сохранение результатов в Excel (в папке проекта)."""
    out_path = os.path.join(out_dir, "sa_results_variant1.xlsx")
    best_trace_df = best_result["trace"].copy()
    summary_rows = []
    for r in all_results:
        summary_rows.append(
            {
                "запуск": int(r["run"]),
                "x1": float(r["x"][0]),
                "x2": float(r["x"][1]),
                "x3": float(r["x"][2]),
                "f": float(r["f"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("f").reset_index(drop=True)
    params_df = pd.DataFrame(list(params.items()), columns=["параметр", "значение"])

    os.makedirs(out_dir, exist_ok=True)
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            best_trace_df.to_excel(writer, sheet_name="трасса_лучшего_прогона", index=False)
            summary_df.to_excel(writer, sheet_name="сводка_всех_запусков", index=False)
            params_df.to_excel(writer, sheet_name="параметры", index=False)

            for sheet_name, df in [("трасса_лучшего_прогона", best_trace_df), ("сводка_всех_запусков", summary_df), ("параметры", params_df)]:
                worksheet = writer.sheets[sheet_name]
                for i, col in enumerate(df.columns):
                    try:
                        max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    except Exception:
                        max_len = len(col) + 2
                    worksheet.set_column(i, i, max_len)
    except Exception as e:
        raise e

    return out_path

def plot_convergence(best_trace_df: pd.DataFrame, out_dir: str):
    png_path = os.path.join(out_dir, "sa_convergence_variant1.png")
    plt.figure(figsize=(8, 5))
    plt.plot(best_trace_df["итерация"].values, best_trace_df["f"].values)
    plt.xlabel("Итерация")
    plt.ylabel("f(x)")
    plt.title("Сходимость f по итерациям (лучший прогон)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    return png_path

class SAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Имитация отжига — лабораторная (вариант 1)")
        self.geometry("900x700")

        try:
            project_dir = os.path.abspath(os.path.dirname(__file__))
            if not project_dir:
                project_dir = os.getcwd()
        except Exception:
            project_dir = os.getcwd()
        self.project_dir = project_dir

        self.log_queue = queue.Queue()

        self.create_widgets()
        self.poll_log_queue()

    def create_widgets(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill=tk.X, padx=8, pady=6)

        param_frame = ttk.LabelFrame(frm_top, text="Параметры алгоритма")
        param_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        labels = [
            ("Число запусков (n_runs):", "n_runs"),
            ("Начальная температура (T0):", "T0"),
            ("Минимальная температура (Tmin):", "Tmin"),
            ("Коэффициент охлаждения (alpha):", "alpha"),
            ("Попыток на T (attempts_per_T):", "attempts_per_T"),
            ("Шкала шага (step_scale):", "step_scale"),
            ("Макс. итераций (max_iter):", "max_iter"),
            ("Seed (seed):", "seed"),
        ]
        defaults = {
            "n_runs": "6",
            "T0": "50.0",
            "Tmin": "1e-5",
            "alpha": "0.92",
            "attempts_per_T": "300",
            "step_scale": "0.25",
            "max_iter": "20000",
            "seed": "2025",
        }
        self.param_vars = {}
        for i, (text, key) in enumerate(labels):
            lbl = ttk.Label(param_frame, text=text)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=4, pady=2)
            var = tk.StringVar(value=defaults[key])
            ent = ttk.Entry(param_frame, textvariable=var, width=12)
            ent.grid(row=i, column=1, sticky=tk.W, padx=4, pady=2)
            self.param_vars[key] = var

        btn_frame = ttk.Frame(frm_top)
        btn_frame.pack(side=tk.RIGHT, padx=8)
        self.btn_run = ttk.Button(btn_frame, text="Запустить", command=self.on_run)
        self.btn_run.pack(fill=tk.X, pady=3)
        self.btn_stop = ttk.Button(btn_frame, text="Остановить (прервать)", command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=3)
        self.btn_open = ttk.Button(btn_frame, text="Открыть папку результатов", command=self.open_results_folder)
        self.btn_open.pack(fill=tk.X, pady=3)

        log_frame = ttk.LabelFrame(self, text="Лог выполнения")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.txt_log = tk.Text(log_frame, height=18, wrap=tk.NONE)
        self.txt_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_v = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt_log.yview)
        scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log['yscrollcommand'] = scroll_v.set

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=8, pady=6)

        self.progress = ttk.Progressbar(bottom_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        self.lbl_progress = ttk.Label(bottom_frame, text="Готов")
        self.lbl_progress.grid(row=0, column=1, sticky=tk.W, padx=6)

        result_frame = ttk.LabelFrame(self, text="Результат (лучший прогон)")
        result_frame.pack(fill=tk.X, padx=8, pady=6)

        self.result_vars = {
            "x1": tk.StringVar(value=""),
            "x2": tk.StringVar(value=""),
            "x3": tk.StringVar(value=""),
            "f": tk.StringVar(value=""),
            "excel": tk.StringVar(value=""),
            "png": tk.StringVar(value=""),
            "csv": tk.StringVar(value=""),
        }
        ttk.Label(result_frame, text="x1:").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["x1"], width=20, state='readonly').grid(row=0, column=1, padx=2)
        ttk.Label(result_frame, text="x2:").grid(row=0, column=2, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["x2"], width=20, state='readonly').grid(row=0, column=3, padx=2)
        ttk.Label(result_frame, text="x3:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["x3"], width=20, state='readonly').grid(row=1, column=1, padx=2)
        ttk.Label(result_frame, text="f:").grid(row=1, column=2, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["f"], width=20, state='readonly').grid(row=1, column=3, padx=2)

        ttk.Label(result_frame, text="Excel:").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["excel"], width=60, state='readonly').grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=2)
        ttk.Label(result_frame, text="PNG:").grid(row=3, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["png"], width=60, state='readonly').grid(row=3, column=1, columnspan=3, sticky=tk.W, padx=2)
        ttk.Label(result_frame, text="CSV:").grid(row=4, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(result_frame, textvariable=self.result_vars["csv"], width=60, state='readonly').grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=2)

        self.worker_thread = None
        self._stop_event = threading.Event()

    def log(self, text: str):
        """Поместить сообщение в очередь (вызывается из любого потока)."""
        self.log_queue.put(str(text) + "\n")

    def poll_log_queue(self):
        """Периодически вызывать из UI, чтобы подтянуть логи из очереди."""
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.txt_log.insert(tk.END, msg)
                self.txt_log.see(tk.END)
        except queue.Empty:
            pass

        self.after(150, self.poll_log_queue)

    def set_progress(self, value: int, maximum: int):
        """Установить прогресс (вызов из рабочего потока через safe wrapper)."""
        try:
            if maximum <= 0:
                percent = 0
            else:
                percent = int(100 * (value / maximum))
            self.progress['value'] = percent
            self.lbl_progress['text'] = f"{value}/{maximum} ({percent}%)"
        except Exception:
            pass

    def on_run(self):
        try:
            params = {
                "n_runs": int(self.param_vars["n_runs"].get()),
                "T0": float(self.param_vars["T0"].get()),
                "Tmin": float(self.param_vars["Tmin"].get()),
                "alpha": float(self.param_vars["alpha"].get()),
                "attempts_per_T": int(self.param_vars["attempts_per_T"].get()),
                "step_scale": float(self.param_vars["step_scale"].get()),
                "max_iter": int(self.param_vars["max_iter"].get()),
                "seed": int(self.param_vars["seed"].get()),
            }
        except Exception as e:
            messagebox.showerror("Ошибка параметров", f"Неверно введены параметры: {e}")
            return

        # disable controls
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.txt_log.delete(1.0, tk.END)
        self.progress['value'] = 0
        self.lbl_progress['text'] = "Запуск..."

        for k in self.result_vars:
            self.result_vars[k].set("")

        self._stop_event.clear()

        self.worker_thread = threading.Thread(target=self.worker_task, args=(params,), daemon=True)
        self.worker_thread.start()

    def on_stop(self):
        self._stop_event.set()
        self.log("Запрос на прерывание получен. Подождите, идёт корректное завершение...")

    def open_results_folder(self):
        folder = self.project_dir
        try:
            if os.name == 'nt':
                os.startfile(folder)
            elif os.name == 'posix':
                try:
                    if 'darwin' in os.sys.platform:
                        os.system(f'open "{folder}"')
                    else:
                        os.system(f'xdg-open "{folder}"')
                except Exception:
                    messagebox.showinfo("Открытие папки", f"Папка с результатами: {folder}")
            else:
                messagebox.showinfo("Открытие папки", f"Папка с результатами: {folder}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку: {e}")

    def worker_task(self, params):
        """Фоновая задача: запуск нескольких прогонов и сохранение результатов."""
        try:
            n_runs = params["n_runs"]

            param_kwargs = {
                "T0": params["T0"],
                "Tmin": params["Tmin"],
                "alpha": params["alpha"],
                "attempts_per_T": params["attempts_per_T"],
                "step_scale": params["step_scale"],
                "max_iter": params["max_iter"],
            }
            seed = params["seed"]
            out_dir = self.project_dir

            self.log(f"Проектная папка (файлы будут сохранены здесь): {out_dir}")
            self.log(f"Запуск {n_runs} прогонов имитации отжига...")

            # Подготовим прогресс callback — будем обновлять UI через after
            total_estimated = n_runs * param_kwargs.get("max_iter", 20000)

            # progress_cb получает глобальные значения и обновляет прогрессбар реже
            # Обратите внимание: не логируем каждый прогресс, чтобы не засорять очередь.
            def progress_cb(global_iter, total):
                try:
                    # обновляем прогрессбар в UI-потоке
                    self.after(0, self.set_progress, global_iter, total)
                except Exception:
                    pass

            def log_cb(s):
                self.log(s)

            # Передаём progress_interval, чтобы simulated_annealing знал как часто вызывать progress_cb
            param_kwargs["progress_interval"] = max(1, param_kwargs.get("max_iter", 20000) // 1000)
            param_kwargs["log_interval"] = max(1, param_kwargs.get("max_iter", 20000) // 200)  # иногда писать лог

            all_results, best = run_multiple_runs(
                n_runs=n_runs,
                seed=seed,
                param_kwargs=param_kwargs,
                init_bounds=(-1.5, 2.0),
                progress_callback=progress_cb,
                log_callback=log_cb
            )

            if self._stop_event.is_set():
                self.log("Выполнение было прервано пользователем.")
                self.after(0, self.finish_run, None, None, None)
                return

            try:
                excel_path = save_results_to_excel(out_dir, all_results, best, params)
                csv_path = os.path.join(out_dir, "sa_trace_variant1_best.csv")
                best["trace"].to_csv(csv_path, index=False)
                png_path = plot_convergence(best["trace"], out_dir)
                self.log("Файлы сохранены:")
                self.log(f" - Excel: {excel_path}")
                self.log(f" - CSV: {csv_path}")
                self.log(f" - PNG: {png_path}")
            except Exception as e:
                self.log(f"Ошибка при сохранении файлов: {e}")
                excel_path = csv_path = png_path = ""

            x = best["x"]
            fval = best["f"]
            self.after(0, self.finish_run, x, fval, {"excel": excel_path, "csv": csv_path, "png": png_path})
            self.log("Готово.")
        except Exception as ex:
            self.log(f"Выполнение завершилось с ошибкой: {ex}")
            self.after(0, self.finish_run, None, None, None)
        finally:
            self.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            self.after(0, lambda: self.btn_stop.config(state=tk.DISABLED))
            self.after(0, lambda: self.set_progress(0, 1))

    def finish_run(self, x, fval, paths):
        """Вывести результаты в форму (выполняется в главном потоке)."""
        if x is None:
            self.result_vars["x1"].set("")
            self.result_vars["x2"].set("")
            self.result_vars["x3"].set("")
            self.result_vars["f"].set("")
            self.result_vars["excel"].set("")
            self.result_vars["csv"].set("")
            self.result_vars["png"].set("")
            self.lbl_progress['text'] = "Завершено (ошибка/прервано)"
            return

        self.result_vars["x1"].set(f"{float(x[0]):.9g}")
        self.result_vars["x2"].set(f"{float(x[1]):.9g}")
        self.result_vars["x3"].set(f"{float(x[2]):.9g}")
        self.result_vars["f"].set(f"{float(fval):.9g}")
        if paths:
            self.result_vars["excel"].set(paths.get("excel", ""))
            self.result_vars["csv"].set(paths.get("csv", ""))
            self.result_vars["png"].set(paths.get("png", ""))
        self.lbl_progress['text'] = "Готов"

def main():
    app = SAApp()
    app.mainloop()

if __name__ == "__main__":
    main()
