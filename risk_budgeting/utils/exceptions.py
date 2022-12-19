class RiskBudgeting(Exception):
    pass

class RiskMeasureNotDefined(RiskBudgeting):
    def __init__(self, data = None):
        message = f"{data} are not define like correct Risk Measure value"
        super().__init__(message)

class BudgetsNotDefined(RiskBudgeting):
    def __init__(self, data = None):
        message = f"{data} are not define like correct budgets value"
        super().__init__(message)