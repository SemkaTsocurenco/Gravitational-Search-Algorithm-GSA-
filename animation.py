import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 1) Загрузка данных
df = pd.read_csv('./data.csv')

# Список уникальных итераций и частиц
iterations = df['iteration']
fitness  = df['fitness']

# # 2) Настройка фигуры для анимации
fig, ax = plt.subplots()
scat = ax.scatter([], [])
ax.set_xlabel('iteration')
ax.set_ylabel('fitness')
ax.set_xlim(-1, 10)
scat.set_offsets(df[['iteration', 'fitness']].values)
plt.plot(df['iteration'], df['fitness'], label=f'aa')
plt.grid(True)

plt.savefig("./plot.png")
plt.show()




df = pd.read_csv('./positions.csv')

# Список уникальных итераций и частиц
iterations = sorted(df['iteration'].unique())
particles  = df['particle'].unique()

# 2) Настройка фигуры для анимации
fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], c='blue', label='Частицы')
best_point_scatter = ax.scatter([], [], c='red', s=100, label='Лучшее решение')
ax.set_xlim(-110, 110)
ax.set_ylim(-110, 110)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Анимация GSA на задаче поиска минимума f(x,y)=x²+y²")
iteration_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def init():
    iteration_text.set_text("")
    return scat, best_point_scatter, iteration_text

def update(frame):
    data = df[df['iteration'] == frame]
    points = data[['x0', 'x1']].values
    scat.set_offsets(points)
    best_points = data[[ 'bestx0', 'bestx1']].values
    best_point_scatter.set_offsets(best_points)
    ax.set_title(f'Iteration {frame}')
    iteration_text.set_text(f"Итерация: {frame+1}")
    return scat, best_point_scatter, iteration_text


ani = FuncAnimation(
    fig, update, frames=iterations,
    init_func=init, blit=True,
    interval=100, repeat=True
)

plt.show()