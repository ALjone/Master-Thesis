import torch
from game import snake_env
import numpy as np
from tkinter import *

class Graphics:
    def __init__(self, size) -> None:
        self.root = Tk()
        self.size = size
        block_width = int(800/size)
        block_height = int(800/size)
        width = block_width*size
        height = block_height*size
        self.canvas = Canvas(self.root, width=width, height=height)

        self.rects = {}
        for x in range(size):
            for y in range(size):
                fill = "black"
                self.rects[(x, y)] = self.canvas.create_rectangle(y*block_width, x*block_height, (y+1)*block_width, (x+1)*block_height, fill=fill)

        self.canvas.focus_set()
        self.canvas.pack()

        self.root.focus_set()
        #self.root.mainloop()


    def updateWin(self, game_map: snake_env, reward):
        self.root.title(f"Total reward: {reward}")
        for x in range(self.size):
                for y in range(self.size):
                    color = self.canvas.itemcget(self.rects[(y, x)], "fill")

                    if color != "black" and torch.sum(game_map[:, x, y]) < 1:
                        self.canvas.itemconfig(self.rects[(y, x)], fill="black")
                    if game_map[0, x, y] == 1 and color != "white":
                        self.canvas.itemconfig(self.rects[(y, x)], fill="white")

                    if color != "grey" and game_map[1, x, y] == 1:
                        self.canvas.itemconfig(self.rects[(y, x)], fill="grey")

                    if game_map[2, x, y] == 1 and color != "red":
                        apple = (x, y)
                        self.canvas.itemconfig(self.rects[(y, x)], fill="red")


        self.root.update()

        