from model import CalculatorModel
from view import CalculatorView
from controller import CalculatorController

def main():
    model = CalculatorModel()
    view = CalculatorView()
    controller = CalculatorController(model, view)
    controller.run()

if __name__ == "__main__":
    main()
