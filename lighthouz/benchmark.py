import base64
import glob
import os
<<<<<<< HEAD
from typing import Any, List, Literal
=======
from typing import List, Literal, Any, Optional
>>>>>>> f8c6b6d (updated backend API calls)

import requests
from marshmallow import ValidationError

from lighthouz import Lighthouz
from lighthouz.schema import benchmark_schema


class Benchmark:
    def __init__(self, LH: Lighthouz):
        self.LH = LH

<<<<<<< HEAD
    def generate_rag_benchmark_from_file(self, file_path: str):
        if not file_path.endswith(".pdf"):
            return {"success": False, "message": "Only PDF files are supported"}
        if not os.path.isfile(file_path):
            return {"success": False, "message": "File does not exist"}
        with open(file_path, "rb") as pdf_file:
            pdf_data = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")

        url = f"{self.LH.base_url}/api/generate_benchmark"
        data = {
            "input": pdf_base64,
            #   "benchmarks": ["rag_benchmark"],
            "benchmarks": [
                "rag_benchmark",
                "out_of_context",
                "prompt_injection",
                "pii_leak",
            ],
            "filename": os.path.basename(file_path),
        }
        headers = {
            "api-key": self.LH.lh_api_key,
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            benchmark_id = response.json()["benchmark_id"]
            return {"success": True, "benchmark_id": benchmark_id}
        else:
            return {"success": False, "message": response.json()}

    def generate_rag_benchmark_from_folder(self, folder_path: str):
        if os.path.isdir(folder_path):
            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
            if len(pdf_files) == 0:
                return {"success": False, "message": "No PDF files found in folder"}
        else:
            return {"success": False, "message": "Folder does not exist"}

        inputs = []
        for file_path in pdf_files:
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
                inputs.append(
                    {
                        "input": pdf_base64,
                        "filename": os.path.basename(file_path),
                    }
                )
        print("Generating benchmark for {} files".format(len(inputs)))
        response = self.generate_rag_benchmark_from_file(pdf_files[0])
        if not response.get("success", False):
            return response
        print(f"Generated benchmark for 1 file: {pdf_files[0]}")
        benchmark_id = response["benchmark_id"]
        for i in range(1, len(inputs)):
            url = f"{self.LH.base_url}/api/docqa_generate/{benchmark_id}"
            data = inputs[i]
=======
    def generate_benchmark(self, benchmarks: List[str], file_path: Optional[str] = None, folder_path: Optional[str] = None):
        print("Generating benchmark. This might take a few minutes.")
        if not file_path and not folder_path:
            url = f"{self.LH.base_url}/benchmarks/generate"
            data = {"benchmarks": benchmarks}
>>>>>>> f8c6b6d (updated backend API calls)
            headers = {
                "api-key": self.LH.lh_api_key,
            }

<<<<<<< HEAD
=======
            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                benchmark_id = response.json()["benchmark_id"]
                print("Success! The benchmark has been created with id: ", benchmark_id)
                return {"success": True, "benchmark_id": benchmark_id}
            else:
                print("An error has occurred: ", response.json())
                return {"success": False, "message": response.json()}

        if file_path: 
            if not file_path.endswith(".pdf"):
                print("Only PDF files are supported")
                return {"success": False, "message": "Only PDF files are supported"}
            if not os.path.isfile(file_path):
                print("File does not exist")
                return {"success": False, "message": "File does not exist"}
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
            
            benchmarks.append("rag_benchmark")

            url = f"{self.LH.base_url}/benchmarks/generate"
            data = {"input": pdf_base64, 
                "benchmarks": benchmarks, 
                "filename": os.path.basename(file_path)}
            headers = {
                "api-key": self.LH.lh_api_key,
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                benchmark_id = response.json()["benchmark_id"]
                print("Success! The benchmark has been created with id: ", benchmark_id)
                return {"success": True, "benchmark_id": benchmark_id}
            else:
                print("An error has occurred: ", response.json())
                return {"success": False, "message": response.json()}
        
        if folder_path: 
            if os.path.isdir(folder_path):
                pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
                if len(pdf_files) == 0:
                    print("No PDF files found in the folder.")
                    return {"success": False, "message": "No PDF files found in the folder"}
            else:
                print("Folder does not exist")
                return {"success": False, "message": "Folder does not exist"}

            inputs = []
            for file_path in pdf_files:
                with open(file_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
                    inputs.append(
                        {
                            "input": pdf_base64,
                            "benchmarks": ["rag_benchmark"],
                            "filename": os.path.basename(file_path),
                        }
                    )
            print("Generating benchmark with {} files".format(len(inputs)))
            
            benchmarks.append("rag_benchmark")

            url = f"{self.LH.base_url}/benchmarks/generate"
            data = {"input": inputs[0].input, 
                "benchmarks": benchmarks, 
                "filename": inputs[0].filename}
            headers = {
                "api-key": self.LH.lh_api_key,
            }

            response = requests.post(url, headers=headers, json=data)

            if not response.get("success", False):
                return response
            
            print(f"Processed file #1: {pdf_files[0]}")
            benchmark_id = response["benchmark_id"]
            for i in range(1, len(inputs)):
                url = f"{self.LH.base_url}/benchmarks/generate/{benchmark_id}"
                data = inputs[i]
                headers = {
                    "api-key": self.LH.lh_api_key,
                }
                response = requests.put(url, headers=headers, json=data)
                if response.status_code != 200:
                    print("An error has occurred: ", response.json())
                    return {"success": False, "message": response.json()}
                print(f"Processed file #{i + 1}: {pdf_files[i]}")
            
            print("Success! The benchmark has been created with id: ", benchmark_id)
            return {"success": True, "benchmark_id": benchmark_id}

    
    def extend_benchmark(self, benchmark_id: str, benchmarks: List[str], file_path: Optional[str] = None, folder_path: Optional[str] = None):
        print("Updating benchmark. This might take a few minutes.")
        if not file_path and not folder_path:
            url = f"{self.LH.base_url}/benchmarks/generate/{benchmark_id}"
            data = {"benchmarks": benchmarks}
            headers = {
                "api-key": self.LH.lh_api_key,
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                benchmark_id = response.json()["benchmark_id"]
                print("Success! The benchmark has been updated. The id is still the same: ", benchmark_id)
                return {"success": True, "benchmark_id": benchmark_id}
            else:
                print("An error has occurred: ", response.json())
                return {"success": False, "message": response.json()}

        if file_path: 
            if not file_path.endswith(".pdf"):
                print("Only PDF files are supported")
                return {"success": False, "message": "Only PDF files are supported"}
            if not os.path.isfile(file_path):
                print("File does not exist")
                return {"success": False, "message": "File does not exist"}
            with open(file_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
            
            benchmarks.append("rag_benchmark")

            url = f"{self.LH.base_url}/benchmarks/generate/{benchmark_id}"
            data = {"input": pdf_base64, 
                "benchmarks": benchmarks, 
                "filename": os.path.basename(file_path)}
            headers = {
                "api-key": self.LH.lh_api_key,
            }

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                benchmark_id = response.json()["benchmark_id"]
                print("Success! The benchmark has been updated. The id is still the same: ", benchmark_id)
                return {"success": True, "benchmark_id": benchmark_id}
            else:
                print("An error has occurred: ", response.json())
                return {"success": False, "message": response.json()}
        
        if folder_path: 
            if os.path.isdir(folder_path):
                pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
                if len(pdf_files) == 0:
                    print("No PDF files found in the folder.")
                    return {"success": False, "message": "No PDF files found in the folder"}
            else:
                print("Folder does not exist")
                return {"success": False, "message": "Folder does not exist"}

            inputs = []
            for file_path in pdf_files:
                with open(file_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                    pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
                    inputs.append(
                        {
                            "input": pdf_base64,
                            "benchmarks": ["rag_benchmark"],
                            "filename": os.path.basename(file_path),
                        }
                    )
            print("Generating benchmark with {} files".format(len(inputs)))
            
            benchmarks.append("rag_benchmark")

            url = f"{self.LH.base_url}/benchmarks/generate/{benchmark_id}"
            data = {"input": inputs[0].input, 
                "benchmarks": benchmarks, 
                "filename": inputs[0].filename}
            headers = {
                "api-key": self.LH.lh_api_key,
            }

            response = requests.post(url, headers=headers, json=data)

            if not response.get("success", False):
                return response
            
            print(f"Processed file #1: {pdf_files[0]}")
            benchmark_id = response["benchmark_id"]
            for i in range(1, len(inputs)):
                url = f"{self.LH.base_url}/benchmarks/generate/{benchmark_id}"
                data = inputs[i]
                headers = {
                    "api-key": self.LH.lh_api_key,
                }
                response = requests.put(url, headers=headers, json=data)
                if response.status_code != 200:
                    print("An error has occurred: ", response.json())
                    return {"success": False, "message": response.json()}
                print(f"Processed file #{i + 1}: {pdf_files[i]}")
            
            print("Success! The benchmark has been updated. The id is still the same: ", benchmark_id)
            return {"success": True, "benchmark_id": benchmark_id}
    
>>>>>>> f8c6b6d (updated backend API calls)
    def upload_benchmark(
        self,
        benchmark_name: str,
        benchmark_type: Literal["RAG chatbot", "non-Rag chatbot"],
        puts: List[dict[str, Any]],
    ):
        for put in puts:
            try:
                benchmark_schema.load(put)
            except ValidationError as e:
                return {"success": False, "message": str(e)}
        url = f"{self.LH.base_url}/benchmarks/upload"
        headers = {
            "api-key": self.LH.lh_api_key,
        }
        data = {
            "name": benchmark_name,
            "benchmark_type": benchmark_type,
            "benchmark": puts,
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            benchmark_id = response.json()["benchmark_id"]
            print("Success! The benchmark has been uploaded and assigned an id: ", benchmark_id)
            return {"success": True, "benchmark_id": benchmark_id}
        else:
            return {"success": False, "message": response.json()}

    
