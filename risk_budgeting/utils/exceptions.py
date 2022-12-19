from risk_budgeting.config._settings import BUDGETS

class RiskBudgeting(Exception):
    pass

class RiskMeasureNotDefined(RiskBudgeting):
    def __init__(self, data = None):
        message = f"{data} are not define like correct Risk Measure value"
        super().__init__(message)

class BudgetsNameNotRecognize(RiskBudgeting):
    def __init__(self, data = None):
        message = f"{data} are not define like correct budgets name. All budgets name avalaible are : {BUDGETS}"
        super().__init__(message)

class BudgetsNameMissing(RiskBudgeting):
    def __init__(self):
        message = "The budget.name must be define."
        super().__init__(message)

class BudgetsValueSizeNotCorrect(RiskBudgeting):
    def __init__(self, data = None):
        message = f"The budgets value : {data} should be in the range (0,1)."
        super().__init__(message)

class BetaSizeNotCorrect(RiskBudgeting):
    def __init__(self, data = None):
        message = f"Beta : {data} should greater than 0.'"
        super().__init__(message)