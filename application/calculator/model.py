import math

class CalculatorModel:
    def evaluate_expression(self, expression):
        """
        计算表达式的值。
        """
        try:
            # 允许的字符集不再仅仅是数字和简单运算符，因为我们引入了 math 函数
            # 这里我们采用更灵活的方式：使用 eval 的安全上下文
            
            if not expression:
                return ""

            # 替换一些符号以适应 Python 语法
            # 例如将 ^ 替换为 ** (幂运算)
            expression = expression.replace('^', '**')
            
            # 创建安全的执行环境
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            allowed_names['abs'] = abs
            allowed_names['round'] = round
            allowed_names['ln'] = math.log
            
            # 使用 eval 计算结果
            result = str(eval(expression, {"__builtins__": None}, allowed_names))
            return result
        except Exception:
            return "Error"
