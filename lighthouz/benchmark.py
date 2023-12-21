import base64
import glob
import os

import requests

from lighthouz import Lighthouz


class Benchmark:
    def __init__(self, LH: Lighthouz):
        self.LH = LH

    def generate_rag_benchmark_from_file(self, file_path: str):
        if not file_path.endswith(".pdf"):
            return {"success": False, "message": "Only PDF files are supported"}
        if not os.path.isfile(file_path):
            return {"success": False, "message": "File does not exist"}
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

    def generate_rag_benchmark_from_folder(self, folder_path: str):
        if os.path.isdir(folder_path):
            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
            if len(pdf_files) == 0:
                return {"success": False, "message": "No PDF files found in folder"}
        else:
            return {"success": False, "message": "Folder does not exist"}

        benchmarks = {
            "success": True,
            "results": {"benchmark": []},
        }
        for pdf_file in pdf_files:
            response = self.generate_rag_benchmark_from_file(pdf_file)
            if not response["success"]:
                return response
            else:
                benchmarks["results"]["benchmark"] += response["results"]["benchmark"]
        return benchmarks
