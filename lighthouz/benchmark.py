import base64

import requests


class Benchmark:
    def __init__(self, LH):
        self.LH = LH

    def generate_rag_benchmark(self, file_path: str):
        # Read the PDF file and convert it to base64
        with open(file_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

        url = f"{self.LH.base_url}/api/docqa_generate"
        data = {"input": pdf_base64}
        headers = {
            "api-key": self.LH.lh_api_key,
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": response.json()}
