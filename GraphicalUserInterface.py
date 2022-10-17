from tkinter import *


class InputBox:


    def __init__(self, window, xCoordinate, yCoordinate, width, height, bgColor, fgColor, border, font=('Calibri', 12)) -> None:
        self.window = window
        self.Box = Text(self.window, width = width, height=height, border=border, bg=bgColor, fg = fgColor, font = font)
        self.Box.place(x = xCoordinate, y = yCoordinate)


    def GetText(self):
        return self.Box.get('1.0', END).rstrip()


class LabelBox:


    def __init__(self, window, xCoordinate, yCoordinate, text, bgColor, fgColor, font = ('Calibri', 12)) -> None:
        self.Box = Label(window, text = text, bg = bgColor, fg = fgColor, font = font)
        self.Box.place(x = xCoordinate, y = yCoordinate)


    def SetText(self, text):
        self.Box.config(text=text)


class ButtonBox:
    

    def __init__(self, window, xCoordinate, yCoordinate, text, bgColor, fgColor, activeBackground, activeForeground, command, *args) -> None:
        self.Button = Button(window, text= text, bg = bgColor, fg = fgColor, activebackground=activeBackground, activeforeground=activeForeground, command=lambda:command(*args))
        self.Button.place(x = xCoordinate, y = yCoordinate)


class RadioBox:


    def __init__(self, window, xCoordinate, yCoordinate, text, value, variable, bgColor, fgColor) -> None:
        self.RadioButton = Radiobutton(master = window, text = text, value = value, variable=variable, background=bgColor, foreground=fgColor, activebackground=bgColor, activeforeground=fgColor)
        self.RadioButton.place(x = xCoordinate, y = yCoordinate)
