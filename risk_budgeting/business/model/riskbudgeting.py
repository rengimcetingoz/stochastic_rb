from pydantic import BaseModel, StrictBool, StrictFloat, StrictStr, validator
import numpy as np

from risk_budgeting.config._settings import RISK_MEASURE, BUDGETS
from risk_budgeting.utils.exceptions import RiskMeasureNotDefined, BudgetsNameNotRecognize, BudgetsNameMissing


class Budgets(BaseModel):
    name : StrictStr = None
    value : np.ndarray = np.zeros(shape=(0, 0))
    
    @validator('name', pre=True, always=True)
    def check_budgets(cls, v):
        if v is None : raise BudgetsNameMissing()
        if v.upper() not in BUDGETS: raise BudgetsNameNotRecognize(v)
        return v

    class Config:
        arbitrary_types_allowed = True

class RiskBudgetingParams(BaseModel):
    risk_measure : StrictStr ='volatility'
    budgets : Budgets
    expectation : bool =False
    beta : StrictFloat =1.00
    delta : StrictFloat =1.00
    alpha : StrictFloat = None
    gamma : StrictFloat = None

    @validator('risk_measure')
    def check_risk_measure(cls, v, values, **kwargs):
        if v not in RISK_MEASURE: raise RiskMeasureNotDefined(v)
        return v
  #  alpha=None
  #  gamma=None