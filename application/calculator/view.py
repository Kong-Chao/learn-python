import tkinter as tk
from tkinter import ttk

class CalculatorView(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python MVC Calculator")
        self.geometry("400x500")
        self.resizable(False, False)
        
        # 结果显示变量
        self.display_var = tk.StringVar()
        self._create_display()
        self._create_buttons()

    def _create_display(self):
        display_frame = ttk.Frame(self)
        display_frame.pack(expand=True, fill="both")
        
        entry = ttk.Entry(display_frame, textvariable=self.display_var, justify="right", font=("Arial", 20))
        entry.pack(expand=True, fill="both", padx=5, pady=5)

    def _create_buttons(self):
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(expand=True, fill="both")
        
        buttons = [
            ('sin', 0, 0), ('cos', 0, 1), ('tan', 0, 2), ('^', 0, 3), ('C', 0, 4),
            ('(', 1, 0), (')', 1, 1), ('sqrt', 1, 2), ('/', 1, 3), ('*', 1, 4),
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2), ('-', 2, 3), ('ln', 2, 4),
            ('4', 3, 0), ('5', 3, 1), ('6', 3, 2), ('+', 3, 3), ('pi', 3, 4),
            ('1', 4, 0), ('2', 4, 1), ('3', 4, 2), ('=', 4, 3), ('e', 4, 4),
            ('0', 5, 0), ('.', 5, 1), ('DEL', 5, 2), ('%', 5, 3), ('abs', 5, 4)
        ]
        
        # 配置网格权重
        for i in range(6): # 6 rows
            buttons_frame.rowconfigure(i, weight=1)
        for i in range(5): # 5 cols
            buttons_frame.columnconfigure(i, weight=1)

        self.button_widgets = {}
        for text, row, col in buttons:
            btn = ttk.Button(buttons_frame, text=text)
            btn.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            self.button_widgets[text] = btn

    def set_display(self, value):
        self.display_var.set(value)

    def get_display(self):
        return self.display_var.get()
    
    def bind_button(self, button_text, callback):
        if button_text in self.button_widgets:
            self.button_widgets[button_text].config(command=callback)

    def bind_key(self, callback):
        self.bind("<Key>", callback)
