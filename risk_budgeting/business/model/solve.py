from pydantic import BaseModel
import numpy as np
from typing import Optional

class Discretize(BaseModel):
    step : int
    bounds : np.ndarray
    
    class Config:
        arbitrary_types_allowed = True

class SolveParams(BaseModel):
    X : np.ndarray 
    epochs: Optional[int]
    minibatch_size : Optional[int]
    y_init : Optional[np.ndarray]
    t_init : Optional[float]
    eta_0_y : Optional[float]
    eta_0_t : Optional[float]
    c : Optional[float]
    polyak_ruppert:Optional[float]
    discretize:Optional[Discretize]
    proj_y:Optional[np.ndarray]
    store:Optional[bool]
    y_sum : np.ndarray
    sum_k_first : int
    k : int
    y:Optional[np.ndarray]
    n : int
    d : int
    k : int

    class Config:
        arbitrary_types_allowed = True