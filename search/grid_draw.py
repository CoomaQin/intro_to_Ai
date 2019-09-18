from tkinter import *
from matrix import generate_maze


def draw_grid(board, colors, startx=50, starty=50, cellwidth=50):
    width = 2 * startx + len(board) * cellwidth
    height = 2 * starty + len(board) * cellwidth
    canvas.config(width=width, height=height)
    for i in range(len(board)):
        for j in range(len(board)):
            index = board[i][j]
            color = colors[index]
            cellx = startx + i * 50
            celly = starty + j * 50
            canvas.create_rectangle(cellx, celly, cellx + cellwidth, celly + cellwidth,
                                    fill=color, outline="black")
    canvas.create_text([startx + 20, starty + 20], text='s')
    canvas.create_text([width - startx - 20, height - starty - 20], text='g')
    canvas.update()


def reflash():
    ran_board = generate_maze(0.2, 15)
    draw_grid(ran_board, colors)


master = Tk()
canvas = Canvas(master, bg="white")
canvas.pack()
colors = ['white', 'gray', 'black']

btn = Button(master, text="start", command=reflash)
btn.pack()
master.mainloop()
