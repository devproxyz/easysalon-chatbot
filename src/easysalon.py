from dataclasses import dataclass
from typing import Dict, Any

from . import api_services


@dataclass
class Easysalon:

    def __init__(self, base_url: str = "https://eoa.easysalon.vn/api/v1", api_key: str = None):
        """
        Initialize Easysalon Business API connection
        """
        self.base_url = base_url
        self.api_key = api_key
        self.api_service = api_services.APIService(self.base_url, self.api_key)

    def get_salon_info(self):
        return self.api_service._get("info")

    def get_branches(self):
        return self.api_service._get("branchs")

    def get_services(self):
        return self.api_service._get("services")

    def get_products(self):
        return self.api_service._get("products")

    def get_packages(self):
        return self.api_service._get("packages")

    def book_appointment(self, booking_request: Dict[str, Any]) -> Dict[str, Any]:
        """Book an appointment."""
        payload = {
            "customerMobile": booking_request.get("customer_mobile", ""),  # required
            "customerName": booking_request.get("customer_name", ""),  # required
            "totalCustomer": booking_request.get("total_customer", 1),  # required
            "branchId": booking_request.get("branch_id", 8850),  # required
            "bookingDetails": [
                {"serviceStaffs": [{"serviceId": booking_request.get("service_id", 257170)}]}
            ],
            "bookingDate": booking_request.get("booking_date", ""),  # required
            "bookingTime": booking_request.get("booking_time", ""),  # required
        }

        response = self.api_service._post("booking", data=payload)

        print(f"Booking API Request Payload: {payload}")
        return response