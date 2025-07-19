import requests
from typing import List, Dict, Any
from pydantic import BaseModel

class APIService:
    """Base class for API services."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.default_row_per_page = 99999
        self.common_headers = {
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key
        }

    def _get(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any] | Any:
        """Make a GET request to the API."""
        if params is None:
            params = {}
        params["rowPerPage"] = self.default_row_per_page
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, headers = self.common_headers)
            response.raise_for_status()
            return response.json()["data"]
        except Exception as ex:
            print(str(ex))
            return None

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the API."""
        try:
            response = requests.post(f"{self.base_url}/{endpoint}", json=data, headers=self.common_headers)
            response.raise_for_status()
            return response.json()["data"]
        except Exception as ex:
            print(str(ex))
            return None

