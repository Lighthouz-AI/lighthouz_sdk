from typing import Optional

import requests

from lighthouz import Lighthouz

class App:
    def __init__(self, LH: Lighthouz):
        self.LH = LH
    
    def register(self, model: str = "", name: str = "", description: Optional[str] = "",
               app_type: Optional[str] = "", endpoint: Optional[str] = ""):
        
        url = f"{self.LH.base_url}/apps/create"
        headers = {
            "api-key": self.LH.lh_api_key,
        }
        data = {
            "title": name,
            "model": model,
            "description": description,
            "app_type": app_type,
            "endpoint": endpoint
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            app_id = response.json()["app_id"]
            print("Success! The app has been registered and assigned an id: ", app_id)
            return {"success": True, "app_id": app_id}
        else:
            return {"success": False, "message": response.json()}