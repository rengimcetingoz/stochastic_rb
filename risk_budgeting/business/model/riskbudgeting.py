from pydantic import BaseModel, StrictBool, StrictFloat, StrictStr, validator

from risk_budgeting.config._settings import RISK_MEASURE, BUDGETS
from risk_budgeting.utils.exceptions import RiskMeasureNotDefined, BudgetsNotDefined

class RiskBudgetingParams(BaseModel):
    risk_measure : StrictStr ='volatility'
    budgets : StrictStr ='ERC'
    expectation : bool =False
    beta : StrictFloat =1.00
    delta : StrictFloat =1.00

    @validator('risk_measure')
    def check_risk_measure(cls, v, values, **kwargs):
        if v not in RISK_MEASURE: raise RiskMeasureNotDefined(v)
        return v

    @validator('budgets')
    def check_budgets(cls, v, values, **kwargs):
        if v not in BUDGETS: raise BudgetsNotDefined(v)
        return v
  #  alpha=None
  #  gamma=None