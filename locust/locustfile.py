from locust import HttpUser, task, between
import base64
import random
import os

class GenderClassifierUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.test_image_base64 = self.create_test_image_base64()
    
    def create_test_image_base64(self):
        small_png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        return small_png_base64
    
    @task(3)
    def health_check(self):
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status: {response.status_code}")
    
    @task(10)
    def predict_base64(self):
        payload = {
            "image": self.test_image_base64
        }
        
        with self.client.post("/predict_base64", 
                             json=payload,
                             catch_response=True,
                             headers={"Content-Type": "application/json"}) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("success"):
                        response.success()
                    else:
                        response.failure(f"Prediction failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    response.failure(f"Failed to parse JSON response: {str(e)}")
            else:
                response.failure(f"Prediction request failed with status: {response.status_code}")
    
    @task(2)
    def model_info(self):
        with self.client.get("/model_info", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get("success"):
                        response.success()
                    else:
                        response.failure(f"Model info failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    response.failure(f"Failed to parse JSON response: {str(e)}")
            else:
                response.failure(f"Model info request failed with status: {response.status_code}")
    
    @task(1)
    def predict_batch_simulation(self):
        batch_size = random.randint(2, 5)
        
        for i in range(batch_size):
            payload = {
                "image": self.test_image_base64
            }
            
            with self.client.post("/predict_base64", 
                                 json=payload,
                                 catch_response=True,
                                 headers={"Content-Type": "application/json"},
                                 name="batch_predict") as response:
                if response.status_code != 200:
                    response.failure(f"Batch prediction {i+1} failed with status: {response.status_code}")
                    break
                
                try:
                    result = response.json()
                    if not result.get("success"):
                        response.failure(f"Batch prediction {i+1} failed: {result.get('error', 'Unknown error')}")
                        break
                except Exception as e:
                    response.failure(f"Batch prediction {i+1} - Failed to parse JSON: {str(e)}")
                    break
            
            if i == batch_size - 1:
                response.success()

class HighLoadUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        self.test_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
    
    @task
    def rapid_predictions(self):
        payload = {
            "image": self.test_image_base64
        }
        
        with self.client.post("/predict_base64", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Rapid prediction failed: {response.status_code}")

class ErrorTestingUser(HttpUser):
    wait_time = between(2, 5)
    
    @task
    def test_invalid_image(self):
        payload = {
            "image": "invalid_base64_data"
        }
        
        with self.client.post("/predict_base64", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code in [400, 500]:
                response.success()
            else:
                response.failure(f"Expected error response, got: {response.status_code}")
    
    @task
    def test_missing_image(self):
        payload = {}
        
        with self.client.post("/predict_base64", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 400:
                response.success()
            else:
                response.failure(f"Expected 400 status for missing image, got: {response.status_code}")
    
    @task
    def test_invalid_endpoint(self):
        with self.client.get("/invalid_endpoint", catch_response=True) as response:
            if response.status_code == 404:
                response.success()
            else:
                response.failure(f"Expected 404 for invalid endpoint, got: {response.status_code}")

@task
def stress_test_prediction(self):
    payload = {
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    }
    
    response = self.client.post("/predict_base64", json=payload)
    if response.status_code != 200:
        print(f"Stress test failed: {response.status_code}")

class MixedBehaviorUser(HttpUser):
    wait_time = between(1, 4)
    
    def on_start(self):
        self.test_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
    
    @task(5)
    def normal_prediction(self):
        payload = {"image": self.test_image_base64}
        self.client.post("/predict_base64", json=payload)
    
    @task(2)
    def check_health(self):
        self.client.get("/")
    
    @task(1)
    def get_model_info(self):
        self.client.get("/model_info")
