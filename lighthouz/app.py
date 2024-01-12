from typing import Optional, Literal

import requests

from lighthouz import Lighthouz


class App:
    def __init__(self, LH: Lighthouz):
        self.LH = LH

    def register(
        self,
        name: str,
        model: str,
        app_type: Literal["RAG chatbot", "non-Rag chatbot"] = "RAG chatbot",
        description: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        url = f"{self.LH.base_url}/apps/create"
        headers = {
            "api-key": self.LH.lh_api_key,
        }
        data = {
            "title": name,
            "model": model,
            "description": description,
            "app_type": app_type,
            "endpoint": endpoint,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            app_id = response.json()["app_id"]
            print("Success! The app has been registered and assigned an id: ", app_id)
            return {"success": True, "app_id": app_id}
        elif response.status_code == 401:
            raise Exception(
                "Unauthorized. Check your API key. You can find your API key in the Lighthouz dashboard."
            )
        elif response.status_code == 400:
            raise Exception(response.json().get("msg") or "Bad request.")
        else:
            raise Exception("Something went wrong. Please try again later.")
