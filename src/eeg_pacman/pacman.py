#coding:utf-8
from psychopy import visual, event
import numpy as np
from psychopy.visual.pie import Pie
import random

class Pacman():
    def __init__(self, win, gridSize=10, cellSize=0.1, nTyrgets=3):

        self.cellSize = cellSize
        self.rectSize = (gridSize*cellSize, gridSize*cellSize)
        self.rect = visual.Rect(win, size = self.rectSize, lineColor='black')
        self.rect.autoDraw = True

        ballSize = (win.size[::-1] / win.size.max()) * cellSize
        ballPos = -self.rectSize[0] / 2 + cellSize/2
        self.ball = Pie(win, start=-60, end=240.0,
                   size=ballSize, fillColor='yellow',
                   pos=(ballPos, -ballPos), ori=180)
        self.ball.autoDraw = True
        self.horizLines = [visual.line.Line(win, start=(-self.rectSize[0]/2, i), end=(self.rectSize[0]/2, i), lineColor='black')
                           for i in np.arange(-self.rectSize[0]/2 + cellSize, self.rectSize[0]/2, cellSize)]
        self.vertLines = [visual.line.Line(win, start=(i, -self.rectSize[0]/2), end=(i, self.rectSize[0]/2), lineColor='black')
                          for i in np.arange(-self.rectSize[0]/2 + cellSize, self.rectSize[0]/2, cellSize)]
        
        for i in range(len(self.horizLines)):
            self.horizLines[i].autoDraw = True
            self.vertLines[i].autoDraw = True

        #генерируем таргеты
        targets = random.sample(range(2, gridSize**2), nTyrgets)
        coords = np.arange(-self.rectSize[0]/2 + cellSize/2, self.rectSize[0]/2, cellSize)

        targets = [(coords[i % gridSize], -coords[i // gridSize]) for i in targets]
        self.targetCircles = [visual.Circle(win, lineColor='lightgreen', lineWidth=2,
                              fillColor='lightgreen', size=ballSize*0.3, pos=i, autoDraw = True) for i in targets]
        self.activeTargets = nTyrgets

    def chekTarget(self):
        for t in self.targetCircles:
            if t.autoDraw:
                if abs(self.ball.pos[0] - t.pos[0]) < 0.02:
                    if abs(self.ball.pos[1] - t.pos[1]) < 0.02:
                        t.autoDraw = False
                        self.activeTargets -= 1

    def left(self):
        if self.ball.pos[0] > -self.rectSize[0]/2 + self.cellSize:
            self.ball.pos += (-self.cellSize, 0)
            self.ball.ori = 0
            self.chekTarget()

    def right(self):
        if self.ball.pos[0] < self.rectSize[0]/2 - self.cellSize:
            self.ball.pos += (self.cellSize, 0)
            self.ball.ori = 180
            self.chekTarget()

    def up(self):
        if self.ball.pos[1] < self.rectSize[0]/2 - self.cellSize:
            self.ball.pos += (0, self.cellSize)
            self.ball.ori = 90
            self.chekTarget()

    def down(self):
        if self.ball.pos[1] > -self.rectSize[0]/2 + self.cellSize:
            self.ball.pos += (0, -self.cellSize)
            self.ball.ori = 270
            self.chekTarget()

if __name__ == "__main__":
    win = visual.Window()
    p = Pacman(win)
    running = True
    while running:
        move = random.randint(1,4)
        if "escape" in event.getKeys():
            running = False
        if move == 1:
            p.left()
        elif move == 2:
            p.up()
        elif move == 3:
            p.right()
        elif move == 4:
            p.down()

        
        win.flip()
    win.close()