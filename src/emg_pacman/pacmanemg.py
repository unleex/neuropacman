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


def colorchange(stim, color):
    stim.color = color
    stim.draw()

def arm_status(data: np.ndarray, threshold: float, leg_threshold: float):
    stdleft = np.std(data[0])
    stdright = np.std(data[1])
    stdleg = np.std(data[2])
    print(f"{stdleft=} {stdright=} {stdleg=}")
    left_tense = stdleft > threshold
    right_tense = stdright > threshold
    leg_tense = stdleg > leg_threshold
    # left_tense = ((data[0]**2).mean())**0.5 > threshold
    # right_tense = ((data[1]**2).mean())**0.5 > threshold
    print(left_tense, right_tense, leg_tense)
    return left_tense, right_tense, leg_tense

def get_move(left_tense, right_tense, leg_tense):
    if left_tense and not right_tense:
        return "left"
    if right_tense and not left_tense:
        return "right"
    if right_tense and left_tense:
        return "up"
    if leg_tense:
        return "down"
    
def read_arduino(n_channels):
    try:
        line = arduino.readline().decode('utf-8').strip()
        data = list(map(float, line.split(' ')))
        if len(data) == n_channels:
            return data
        print(f"found {len(data)} channels instead of {n_channels}.")
    except Exception as e:
        print(f"Ошибка чтения данных с Arduino: {e}")
        return 
    
emg_std_threshold = 10
leg_std_threshold = 10
n_channels=10 # for arduino to work, we use only first two
left_arm_channel=1 # 0 if A0 and so on
right_arm_channel=4
third_channel = 7
data_required = 8


print("Попытка соединения с Arduino")
try:
    arduino = serial.Serial('COM5', 115200)
    core.wait(2)
    print("Соединение с Arduino установлено.")
except Exception as e:
    print(f"Ошибка подключения к Arduino: {e}")
    core.quit()


# Create a window
win = visual.Window(size=(800, 600), color=(0, 0, 0))
running = True
data: list[list[int]] = [[],[],[]]
pacman = Pacman(win, gridSize=8,nTyrgets=10)
win.flip()
event.waitKeys(keyList=["space"])
clock = core.Clock()
start = clock.getTime()
clock.reset()
while running:
    keys = event.getKeys()
    if 'escape' in keys or pacman.activeTargets == 0:
        running = False
    for _ in range(data_required):
        received = read_arduino(n_channels)
        if received:
            data[0].append(received[left_arm_channel])
            data[1].append(received[right_arm_channel])
            data[2].append(received[third_channel])
    left_tense, right_tense, leg_tense = arm_status(
        np.array(data),
        emg_std_threshold,
        leg_std_threshold
        )
    move = get_move(left_tense, right_tense, leg_tense)
    print(move)
    if move == "left":
        pacman.left()
    elif move == "up":
        pacman.up()
    elif move == "right":
        pacman.right()
    elif move == "down":
        pacman.down()  
    data = [[],[],[]]
    win.flip()

# Close the window
print("time:", clock.getTime(applyZero=True) - start)
win.close()