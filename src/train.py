print("psychoimporting...")
from psychopy import visual, core, event
print("psychoimported!")
import numpy as np
import serial
import scipy.signal as sig
from pacman import Pacman
#coding:utf-8
from psychopy import visual, event
import numpy as np
from psychopy.visual.pie import Pie
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pipeline import pipeline, from_channels


def read_arduino():
    try:
        line = arduino.readline().decode('utf-8').strip()
        data = list(map(float, line.split(' ')))
        return data
    except Exception as e:
        print(f"Ошибка чтения данных с Arduino: {e}")
        return 


def draw_arrows():
    for lines in arrows.values():
        for line in lines:
            line.draw()

def multicolorchange(stims, color):
    for stim in stims:
        stim.color = color

def create_arrow(win, direction, color): 
    arrow_size = 50 
    if direction == 'left': 
        lines = [ 
            visual.Line(win, start=(-0.75, 0), end=(-1, 0), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(-1, 0), end=(-1 + np.sqrt(arrow_size ** 2 / 2) / 400, np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(-1, 0), end=(-1 + np.sqrt(arrow_size ** 2 / 2) / 400, -np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2) 
        ] 
    elif direction == 'right': 
        lines = [ 
            visual.Line(win, start=(0.75, 0), end=(1, 0), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(1, 0), end=(1 - np.sqrt(arrow_size ** 2 / 2) / 400, np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(1, 0), end=(1 - np.sqrt(arrow_size ** 2 / 2) / 400, -np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2) 
        ] 
    elif direction == 'up': 
        lines = [ 
            visual.Line(win, start=(0, 0.5), end=(0, 0.75), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(0, 0.75), end=(-np.sqrt(arrow_size ** 2 / 2) / 400, 0.75 - np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(0, 0.75), end=(np.sqrt(arrow_size ** 2 / 2) / 400, 0.75 - np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2) 
        ] 
    elif direction == 'down': 
        lines = [ 
            visual.Line(win, start=(0, -0.5), end=(0, -0.75), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(0, -0.75), end=(-np.sqrt(arrow_size ** 2 / 2) / 400, -0.75 + np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2), 
            visual.Line(win, start=(0, -0.75), end=(np.sqrt(arrow_size ** 2 / 2) / 400, -0.75 + np.sqrt(arrow_size ** 2 / 2) / 300), lineColor=color, lineWidth=2) 
        ] 
    return lines

def truncate(X, minlen): 
    transformed_X = [] 
    for matrix in X: 
        matrix = np.array(matrix) 
        if matrix.shape[1] > minlen: 
            transformed_matrix = matrix[:, :minlen] 
        else: 
            transformed_matrix = matrix 
        transformed_X.append(transformed_matrix.tolist()) 
    return transformed_X
n_channels=10 # ensure that everything is fine later
eeg_channels = [0,1,2,3,4,5,6,7]


print("Попытка соединения с Arduino")
try:
    arduino = serial.Serial('COM5', 115200)
    core.wait(2)
    print("Соединение с Arduino установлено.")
except Exception as e:
    print(f"Ошибка подключения к Arduino: {e}")
    core.quit()


# Create a window
win = visual.Window(size=(1600, 1200), color=(0.1,)*3)
data: list[list[int]] = [[] for _ in range(len(eeg_channels))]
data_required = 200
arrows = {
    'left': create_arrow(win, 'left', 'white'),
    'right': create_arrow(win, 'right', 'white'),
    'up': create_arrow(win, 'up', 'white'),
    'down': create_arrow(win, 'down', 'white')
}
draw_arrows()
current_arrow_idx = 0
change_to_color = "red"
startcolor = "white"
warning_color = "blue"
Pacman(win, gridSize=7, nTyrgets=10)
X = []
y = []
single_desired_epochs = 2
minlen = data_required + 1
for desired_direction in ["left","right","up","down"] * 3:
    multicolorchange(arrows[desired_direction], warning_color)
    draw_arrows()
    win.flip()
    core.wait(2.0)
    multicolorchange(arrows[desired_direction], startcolor)
    draw_arrows()
    win.flip()
    core.wait(2.0)
    order = ["left","right","up","down"] * single_desired_epochs
    random.shuffle(order)
    for direction in order:
        multicolorchange(arrows[direction], change_to_color)
        draw_arrows()
        win.flip()
        is_target = (desired_direction == direction)
        for _ in range(data_required):
            received = read_arduino()
            if received and len(received) == n_channels:
                for ch in eeg_channels:
                    data[ch].append(received[ch])
        minlen = min(minlen, np.array(data).shape[1])
        X.append(data)
        y.append(is_target)
        data = [[] for _ in range(len(eeg_channels))]
        multicolorchange(arrows[direction], startcolor)
        draw_arrows()
        win.flip()
        core.wait(0.3)       
minlen=195
import numpy as np
X = np.array([from_channels(np.array(i)) for i in truncate(X, minlen)])
np.save("X1", X)
np.save("y1", y)
with open("minlen.txt", "w+") as f:
    f.write(str(minlen))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"{acc=} {f1=}")
with open("pipeline1.pickle", "wb+") as f:
    pickle.dump(pipeline, f)
win.close()