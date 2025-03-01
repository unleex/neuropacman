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

def from_channels(channels):
    channels = np.array(channels)
    return channels.reshape(np.prod(channels.shape))


with open("pipeline.pickle", "rb") as f:
    pipeline = pickle.load(f)
with open("minlen.txt") as f:
    ch_len = int(f.read())
def colorchange(stim, color):
    stim.color = color
    stim.draw()

def get_p300(data: np.ndarray, pipeline):
    # data = np.array([i[eeg_channels] for i in data])
    print(data.shape)
    res = pipeline.predict_proba(data)[:,1]
    print(res)
    return res > 0.4

def get_move(left_tense, right_tense):
    if left_tense:
        if right_tense:
            return "up"
        else:
            return "left"
    else:
        if right_tense:
            return "right"
        else:
            return "down"

def read_arduino(n_channels):
    try:
        line = arduino.readline().decode('utf-8').strip()
        data = list(map(float, line.split(' ')))
        return data
    except Exception as e:
        print(f"Ошибка чтения данных с Arduino: {e}")


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

def multicolorchange(stims, color):
    for stim in stims:
        stim.color = color

def draw_arrows():
    for lines in arrows.values():
        for line in lines:
            line.draw()

n_channels=10 # ensure that everything is fine later
eeg_channels = [0,1,2,3,4,5,6,7]
emg_threshold = 135 # values above count arm as tense
emg_std_threshold = 20

print("Попытка соединения с Arduino")
try:
    arduino = serial.Serial('COM5', 115200)
    core.wait(2)
    print("Соединение с Arduino установлено.")
except Exception as e:
    print(f"Ошибка подключения к Arduino: {e}")
    core.quit()


# Create a window
win = visual.Window(size=(1600, 1200), color=(0, 0, 0))
running = True
data: list[list[int]] = [[] for _ in range(len(eeg_channels))]

data_required = 200
pacman = Pacman(win, gridSize=7,nTyrgets=10)
arrows = {
    'left': create_arrow(win, 'left', 'white'),
    'right': create_arrow(win, 'right', 'white'),
    'up': create_arrow(win, 'up', 'white'),
    'down': create_arrow(win, 'down', 'white')
}
current_arrow_idx = 0
change_to_color = "red"
startcolor = "white"
running = True
win.flip()
event.waitKeys(keyList=["space"])
while running:
    for direction in arrows:
        multicolorchange(arrows[direction], change_to_color)
        draw_arrows()
        win.flip()
        if 'escape' in event.getKeys():
            running = False
            break
        for _ in range(data_required):
            received = read_arduino(n_channels)
            if received and len(received) == n_channels:
                for ch in eeg_channels:
                    data[ch].append(received[ch])
        multicolorchange(arrows[direction], startcolor)
        draw_arrows()
        win.flip()
        is_p300 = get_p300(
            np.array([from_channels(np.array(data)[:,:ch_len])]),
            pipeline
            )[0]
        if is_p300:
            print(direction)
            if direction == "left":
                pacman.left()
            elif direction == "up":
                pacman.up()
            elif direction == "right":
                pacman.right()
            elif direction == "down":
                pacman.down()  
        data = [[] for _ in range(len(eeg_channels))]

# Close the window
win.close()