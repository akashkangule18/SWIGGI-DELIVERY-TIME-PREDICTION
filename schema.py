from pydantic import BaseModel


class DeliveryInput(BaseModel):
    age: float
    rating: float

    weather_cond: str
    traffic: str

    vehicle_cond: int
    vehicle_type: str

    multiple_orders: float

    festival: str
    city_type: str
    city_name: str

    month: int
    weekend: int

    ordered_hour: float
    distance_km: float