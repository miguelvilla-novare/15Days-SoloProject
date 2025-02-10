from typing import Optional, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from commons import init_model

load_dotenv()

model = init_model()


#
# class Joke(BaseModel):
#     setup: str = Field(description="The setup of the joke")
#     punchline: str = Field(description="The punchline to the joke")
#     rating: int = Field(description="How funny the joke is, from 1 to 10.")
#
# structured_output = model.with_structured_output(Joke)
#
# response = structured_output.invoke("Make a programming joke in Python.")
# print(f"Joke: {response.setup}")
# print(f"Punchline: {response.punchline} ({response.rating}/10)")
#

class Receipt(BaseModel):
    total: Optional[float] = Field(None, description="Mark None if unclear")
    items: List[str] = Field(..., description="Best-guess list of items")


receipt_model = model.with_structured_output(Receipt)
kfc_receipt: Receipt = receipt_model.invoke(
    "2-pc Chicken Meal - P230, Ala King Zinger Steak Meal - P180, Zinger - P155")

print("Purchases: ")
for item in kfc_receipt.items:
    print(f"- {item}")

print(f"Total: {kfc_receipt.total}")
