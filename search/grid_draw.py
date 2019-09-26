from tkinter import *
from matrix import generate_maze


# from search_algs import *


def draw_grid(grid, startx=20, starty=20, cellwidth=20):
    master = Tk()
    canvas = Canvas(master, bg="white")
    canvas.pack()
    colors = ['white', 'gray', 'black', 'red', 'white', 'white']
    width = 2 * startx + len(grid) * cellwidth
    height = 2 * starty + len(grid) * cellwidth
    canvas.config(width=width, height=height)
    for i in range(len(grid)):
        for j in range(len(grid)):
            index = grid[i][j]
            color = colors[index]
            cellx = startx + i * 20
            celly = starty + j * 20
            canvas.create_rectangle(cellx, celly, cellx + cellwidth, celly + cellwidth,
                                    fill=color, outline="black")
    canvas.create_text([startx + 10, starty + 10], text='s')
    canvas.create_text([width - startx - 10, height - starty - 10], text='g')
    canvas.update()
    master.mainloop()

# btn = Button(master, text="start", command=reflash)
# btn.pack()
