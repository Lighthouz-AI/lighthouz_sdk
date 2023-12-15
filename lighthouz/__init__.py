class LH:
    def __init__(self, lh_app_id: str, lh_api_key: str):
        self.lh_api_key = lh_api_key
        self.lh_app_id = lh_app_id
        # self.base_url = "https://lighthouz.ai/api"  # Replace with your API endpoint URL
        self.base_url = "http://localhost:5100/"  # Replace with your API endpoint URL
