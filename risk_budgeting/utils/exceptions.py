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

class BudgetsValueSize(RiskBudgeting):
    def __init__(self, data = None):
        message = f"The budgets value : {data} should be in the range (0,1)."
        super().__init__(message)

class BetaSize(RiskBudgeting):
    def __init__(self, data = None):
        message = f"The budgets value : {data} should be in the range (0,1)."
        super().__init__(message)