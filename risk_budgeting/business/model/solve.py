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
    epochs: int = None
    minibatch_size : int= 128
    y_init : np.ndarray= None
    t_init : float= None
    eta_0_y : float= None
    eta_0_t : float= None
    c : float= 0.65
    polyak_ruppert:float= 0.2
    discretize:Discretize= None
    proj_y:np.ndarray= None
    y_sum : np.ndarray= None
    sum_k_first : int= None
    k : int= None
    y:np.ndarray= None
    n : int= None
    d : int= None
    k : int= None

    class Config:
        arbitrary_types_allowed = True