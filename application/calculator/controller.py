from functools import partial

class CalculatorController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self._bind_events()

    def _bind_events(self):
        # 为每个按钮绑定事件
        for btn_text in self.view.button_widgets:
            # 使用 partial 来传递参数，避免闭包问题
            action = partial(self._on_button_click, btn_text)
            self.view.bind_button(btn_text, action)
            
        # 绑定键盘事件
        self.view.bind_key(self._on_key_press)

    def _on_key_press(self, event):
        key = event.char
        keysym = event.keysym
        
        if keysym == 'Return':
            self._on_button_click('=')
        elif keysym == 'BackSpace':
            self._on_button_click('DEL')
        elif keysym == 'Escape':
            self._on_button_click('C')
        elif key:
            # 简单的键盘映射
            if key in '0123456789+-*/().^%':
                self._on_button_click(key)
            elif key == '=':
                self._on_button_click('=')

    def _on_button_click(self, char):
        current_text = self.view.get_display()

        if char == 'C':
            self.view.set_display("")
        elif char == 'DEL':
            self.view.set_display(current_text[:-1])
        elif char == '=':
            result = self.model.evaluate_expression(current_text)
            self.view.set_display(result)
        elif char in ('sin', 'cos', 'tan', 'sqrt', 'ln', 'abs'):
            if current_text == "Error":
                current_text = ""
            self.view.set_display(current_text + char + "(")
        else:
            # 如果当前显示的是 Error，先清空
            if current_text == "Error":
                current_text = ""
            self.view.set_display(current_text + char)

    def run(self):
        self.view.mainloop()
