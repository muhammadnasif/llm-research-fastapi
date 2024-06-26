from langchain.tools import tool, StructuredTool
from pydantic import BaseModel, Field, constr
from datetime import date
from langchain_core.tools import BaseTool
import requests
import datetime
from datetime import date


name = "Shibly"
ROOM_NUMBER = 101
TODAY = date.today()

class OpenMeteoInput(BaseModel):
    latitude: float = Field(...,
                            description="Latitude of the location to fetch weather data for")
    longitude: float = Field(...,
                             description="Longitude of the location to fetch weather data for")
    
class BookRoomInput(BaseModel):
    room_type: str = Field(..., description="Which type of room AC or Non-AC. Input from user")
    class_type: str = Field( ..., description="Which class of room it is. Business class or Economic class.Input from user")
    check_in_date: date = Field(..., description="The date user will check-in. Input from user")
    check_out_date: date = Field(..., description="The date user will check-out. Input from user")
    mobile_no: str = Field(..., description="Mobile number of the user. Input from user")


class HousekeepingServiceEntity(BaseModel):
    reason: str = Field(..., description="The reason for housekeeping service is requested for")

class RoomRecommendation(BaseModel):
    budget_highest: int = Field(..., description="Maximum amount  customer can pay per day for a room.")

class RequestFoodFromRestaurant(BaseModel):
    item_name: str = Field(..., description="The food item customer wants to order from the restaurant")
    item_quantity : int = Field(..., description = "Quantity of food the customer wants to order from the restaurant.")
    dine_in_type: str = Field(..., description="If the customer wants to eat in the door-step, take parcel or dine in the restaurant. It can have at most 3 values 'dine-in-room', 'dine-in-restaurant', 'parcel'")


class TransportationRecommendationEntity(BaseModel):
    location: str = Field(..., description="The place customer wants to go visit")

class RequestBillingChangeRequest(BaseModel):
    complaint: str = Field(..., description="Complain about the bill. It could be that the bill is more than it should be. Or some services are charged more than it was supposed to be")

class RecommendationExcursion(BaseModel):
    place_type: str = Field(
        ..., description="The type of place the customer wants to visit. Example - park, zoo, pool. Take input from user")

class RoomAmenitiesRequest(BaseModel):
    requested_amenity: str = Field(...,description="The amenity that the customer wants to request. Example - towel, pillow, blanket etc. Take input from user.")

class RoomMaintenanceRequestInput(BaseModel):
    issue: str = Field(..., description="The issue for which it needs maintenance service")

class ReminderEntity(BaseModel):
    reminder_message: str = Field(..., description="The reminder message of the customer")
    reminder_date : str = Field(..., description="The date or day to remind. For example : today, tomorrow, 12th October etc.")
    reminder_time: str = Field(..., description="The time to remind at. Time should be in 12 hour format like 4:00PM")

class ShuttleServiceEntity(BaseModel):
    location: str = Field(...,
                          description="The location from where the customer will be picked up ")
    time: str = Field(...,
                      description="The time at which the customer will be picked up")

def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current Weather for given cities or coordinates. For example: what is the weather of Colombo ?"""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(
            f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace(
        'Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(
        time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}Â°C'

def book_room(room_type: str, class_type: str, check_in_date: date, check_out_date: date, mobile_no: str) -> str:
    """
      Book a room with the specified details.

      Args:
          room_type (str): Which type of room to book (AC or Non-AC). Input from user.
          class_type (str): Which class of room it is (Business class or Economic class). Input from user.
          check_in_date (date): The date the user will check-in. Input from user.
          check_out_date (date): The date the user will check-out. Input from user.
          mobile_no (str): Mobile number of the user. Input from user.

      Returns:
          str: A message confirming the room booking.
      """
    # Placeholder logic for booking the room
    # return f"Room has been booked for {room_type} {class_type} class from {check_in_date} to {check_out_date}. Mobile number: {mobile_no}."
    return f'''{{
                  "function-name": "book_room",
                  "parameters": {{
                      "room_type": "{room_type}",
                      "class_type": "{class_type}",
                      "check_in_date": "{check_in_date}",
                      "check_out_date": "{check_out_date}",
                      "mobile_no": "{mobile_no}"
                  }}
              }}'''

def housekeeping_service_request(reason:str) -> str:
    """
    Provides housekeeping service to the hotel room like cleaning.
    """

    return f'''{{
                  "function-name": "housekeeping_service_request",
                  "parameters": {{
                      "room_number": "{ROOM_NUMBER}",
                      "reason" : "{reason}"
                  }}
              }}'''

def room_recommendation(budget_highest: int) -> str:
    """
    Recommend a room for a customer based on his budget which he can pay per day for a room. For example, I want a room that costs less than 1000 per day. 
    Args:
      budget_highest (int) : Maximum rent customer can pay per day for a room. Take input from user
    Returns
      str: A message with room suggestions according to budget.
    """

    return f'''{{
                  "function-name": "room_recommendation",
                  "parameters": {{
                      "budget_highest": "{budget_highest}"
                  }}
              }}'''


def order_restaurant_item(item_name: str, item_quantity:int,  dine_in_type: str) -> str:
    """
    Place order to the restaurant for food items with specified details. For example, I want to order a pizza from the restaurant.

    Args:
      item_name (str) : The food item they want to order from the restaurant
      item_quantity (int) = Quantity of food the customer wants to order from the restaurant
      dine_in_type (str) : inside hotel room, dine in restaurant or parcel

    Returns
      str: A message for confirmation of food order.
    """

    # if dine_in_type == "dine-in-room":
    #     return f"Your order have been placed. The food will get delivered at your room."
    # elif dine_in_type == "dine-in-restaurant":
    #     return f"Your order have been placed. You will be notified once the food is almost ready."
    # else:
    #     return f"Your order have been placed. The parcel will be ready in 45 minutes."

    return f'''{{
                  "function-name": "order_resturant_item",
                  "parameters": {{
                      "item_name": "{item_name}",
                      "item_quantity" : "{item_quantity}",
                      "room_number": "{ROOM_NUMBER}",
                      "dine_in_type": "{dine_in_type}"
                  }}
              }}'''

def bill_complain_request(complaint: str, room_number=ROOM_NUMBER) -> str:
    """
    Complaints about billing with specified details.
    Args:
      complaint (str) : Complain about the bill. It could be that the bill is more than it should be. Or some services are charged more than it was supposed to be.
    Returns
      str: A message for confirmation of the bill complaint.
    """

    # return f"We  have received your complain {complaint}  and notified accounts department to handle the issue. Please keep your patience while we resolve. You will be notified from the front-desk once it is resolved"
    return f'''{{
                  "function-name": "bill_complain_request",
                  "parameters": {{
                      "complaint": "{complaint}",
                      "room_number": "{room_number}"
                  }}
              }}'''


book_room_tool = StructuredTool.from_function(
    func = book_room,
    name = "book_room",
    description = """
      Book a room with the specified details.

      Args:
          room_type (str): Which type of room to book (AC or Non-AC). Input from user.
          class_type (str): Which class of room it is (Business class or Economic class). Input from user.
          check_in_date (date): The date the user will check-in. Input from user.
          check_out_date (date): The date the user will check-out. Input from user.
          mobile_no (str): Mobile number of the user. Input from user.

      Returns:
          str: A message confirming the room booking.
      """,
    args_schema=BookRoomInput,
    return_direct=True,
    infer_schema=True
) 

def transportation_recommendation(location: str) -> str:
    """
    Recommends transportation with specified details
    Args:
      location (str) : The place customer wants to go visit
    Returns
      str: A message with transportation recommendation.
    """

    # transport = "Private Car"

    # return f"I recommend to go there by {transport}"
    return f'''{{
                  "function-name": "transportation_recommendation",
                  "parameters": {{
                      "location": "{location}"
                  }}
              }}'''

def excursion_recommendation(place_type: str) -> str:
    """
    Suggest nice places to visit nearby with specified details
    Args:
      place_type (str) : The type of place the customer wants to visit. Example - park, zoo, pool. Alsways ask for this value.
    Returns
      str: A message with excursion recommendation.
    """

    return f'''
            {{
                "function-name": "excursion_recommendation",
                "parameters": {{
                    "place_type": "{place_type}"
                }}
            }}
            '''

def request_room_amenity(requested_amenity: str):
    """
    Request for room amenities like towel, pillow, blanket etc. Order for room amenities like towel, pillow, blanket etc.

    Args:
      requested_amenity (str) : The amenity that the customer wants to request or order. Example - towel, pillow, blanket etc. Take input from user.
    Returns
      str: An acknowdelgement that ensures that someone is sent to the room for fixing.
    """

    return f'''{{
                  "function-name": "request_room_amenity",
                  "parameters": {{"requested_amenity": "{requested_amenity}",
                  "room_number": "{ROOM_NUMBER}"}}
              }}'''

def request_room_maintenance(issue: str):
    """
    Resolves room issues regarding hardware like toilteries, furnitures, windows or electric gadgets like FAN, TC, AC etc of hotel room.

    Args:
      issue (int) : The issue for which it needs maintenance service
    Returns
      str: An acknowdelgement that ensures that someone is sent to the room for fixing.
    """

    return f'''{{
                  "function-name": "request_room_maintenance",
                  "parameters": {{
                      "issue": "{issue}",
                      "room_number": "{ROOM_NUMBER}"
                  }}
              }}'''


def request_reminder(reminder_message: str, reminder_date:str, reminder_time: str):
    """
    Set an alarm or reminder alarm or reminder call for the customer to remind about the message at the mentioned time.
    For ex,
        Set a meeting tomorrow at 4PM

    Args:
      reminder_message (str) : The reminder message of the customer.
      reminder_date (str) : The day to give the reminder. For example : today, tomorrow, 12th October etc.
      reminder_time (str) : The time to remind the customer at.
    Returns
      str: An acknowdelgement message for the customer.
    """

    return f'''{{
                  "function-name": "request_reminder",
                  "parameters": {{
                        "reminder_date" : "{reminder_date}",
                        "reminder_time": "{reminder_time}",
                        "reminder_message": "{reminder_message}",
                        "room_number": "{ROOM_NUMBER}"
                  }}
              }}'''

def shuttle_service_request(location: str, time: str) -> str:
    """
    Books a shuttle service that picks up or drops off customer.

    Args :
      location (str) : The location from where the customer will be picked up
      time (str) : The exact time of pickup or drop off
    return :
      str : A message that customer is picked or dropped successfully

    """

    return f'''{{
                    "function-name": "shuttle_service_request",
                    "parameters": {{
                        "location": "{location}",
                        "time": "{time}"
                    }}
                }}'''

# Create a structured tool from the function
housekeeping_service_tool = StructuredTool.from_function(
    func=housekeeping_service_request,
    name="housekeeping_service_tool",
    description="Provides housekeeping service to the hotel room like cleaning.",
    args_schema=HousekeepingServiceEntity,
    return_direct=True,
    infer_schema=True
)


room_recommendation_tool = StructuredTool.from_function(
    func=room_recommendation,
    name="room_recommendation_tool",
    description="Recommend a room for a customer based on his budget which he can pay per day for a room.",
    args_schema=RoomRecommendation,
    return_direct=True,
    infer_schema=True
)

order_restaurant_item_tool = StructuredTool.from_function(
    func=order_restaurant_item,
    name="order_restaurant_item_tool",
    description="Place order to the restaurant for food items with specified details.",
    args_schema=RequestFoodFromRestaurant,
    return_direct=True,
    infer_schema=True
)

bill_complain_request_tool = StructuredTool.from_function(
    func=bill_complain_request,
    name="bill_complain_request_tool",
    description="Complaints about billing with specified details.",
    args_schema=RequestBillingChangeRequest,
    return_direct=True,
    infer_schema=True
)


transportation_recommendation_tool = StructuredTool.from_function(
    func=transportation_recommendation,
    name="transportation_recommendation_tool",
    description="Recommends transportation with specified details.",
    args_schema=TransportationRecommendationEntity,
    return_direct=True,
    infer_schema=True
)

excursion_recommendation_tool = StructuredTool.from_function(
    func=excursion_recommendation,
    name="excursion_recommendation_tool",
    description="Suggest nice places to visit nearby with specified details.",
    args_schema=RecommendationExcursion,
    return_direct=True,
    infer_schema=True
)

request_room_amenity_tool = StructuredTool.from_function(
    func=request_room_amenity,
    name="request_room_amenity_tool",
    description="Request for room amenities like towel, pillow, blanket etc. Order for room amenities like towel, pillow, blanket etc.",
    args_schema=RoomAmenitiesRequest,
    return_direct=True,
    infer_schema=True
)

request_room_maintenance_tool = StructuredTool.from_function(
    func=request_room_maintenance,
    name="request_room_maintenance_tool",
    description="Resolves room issues regarding hardware like toiletries, furniture, windows or electric gadgets like FAN, TV, AC etc of hotel room.",
    args_schema=RoomMaintenanceRequestInput,
    return_direct=True,
    infer_schema=True
)

request_reminder_tool = StructuredTool.from_function(
    func=request_reminder,
    name="request_reminder_tool",
    description="Set an alarm or reminder alarm or reminder call for the customer to remind about the message at the mentioned time.",
    args_schema=ReminderEntity,
    return_direct=True,
    infer_schema=True
)

shuttle_service_request_tool = StructuredTool.from_function(
    func=shuttle_service_request,
    name="shuttle_service_request_tool",
    description="Books a shuttle service that picks up or drops off customer.",
    args_schema=ShuttleServiceEntity,
    return_direct=True,
    infer_schema=True
)

get_current_temperature_tool = StructuredTool.from_function(
    func=get_current_temperature,
    name="get_current_temperature_tool",
    description="Fetch current Weather for given cities or coordinates.",
    args_schema=OpenMeteoInput,
    infer_schema=True
)

tools = [
    book_room_tool,
    housekeeping_service_tool,
    room_recommendation_tool,
    order_restaurant_item_tool,
    bill_complain_request_tool,
    transportation_recommendation_tool,
    excursion_recommendation_tool,
    request_room_amenity_tool,
    request_room_maintenance_tool,
    request_reminder_tool,
    shuttle_service_request_tool,
    get_current_temperature_tool

]