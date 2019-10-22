from tkinter import *


def draw_board(board, startx=40, starty=40, cellwidth=40):
    master = Tk()
    canvas = Canvas(master, bg="white")
    canvas.pack()
    values = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '?', 'F']
    width = 2 * startx + len(board) * cellwidth
    height = 2 * starty + len(board) * cellwidth
    canvas.config(width=width, height=height)
    borad_x = board.shape[1]
    borad_y = board.shape[0]
    for i in range(borad_x):
        for j in range(borad_y):
            index = board[j, i]
            value = values[index]
            cellx = startx + i * 40
            celly = starty + j * 40
            if index == 10:
                canvas.create_rectangle(cellx, celly, cellx + cellwidth, celly + cellwidth,
                                        fill='red', outline="black")
            else:
                canvas.create_rectangle(cellx, celly, cellx + cellwidth, celly + cellwidth,
                                        fill='white', outline="black")
            canvas.create_text([cellx+cellwidth/2, celly+cellwidth/2], text=value)
    # canvas.create_text([width - startx - 10, height - starty - 10], text='g')
    canvas.update()
    master.mainloop()
