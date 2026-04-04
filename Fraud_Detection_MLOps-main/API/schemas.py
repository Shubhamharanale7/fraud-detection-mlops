from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

class FraudRequest(BaseModel):
    Category: Optional[str] = Field(None, example="shopping")
    paymentMethod: str = Field(..., example="paypal")
    isWeekend: Optional[int] = Field(None, example=None)
    numItems: int = Field(..., example=4)
    localTime: float = Field(..., example=4.742303)
    paymentMethodAgeDays: int = Field(..., example=0.0)
    accountAgeDays: int = Field(..., example=1)

class FraudResponse(BaseModel):
    prediction: int