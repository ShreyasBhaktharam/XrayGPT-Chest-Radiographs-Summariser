import time
import requests
import numpy as np
import concurrent.futures
import json
from PIL import Image
import base64
import io
import os
import random
import sys

# Configuration
FASTAPI_URL = "http://localhost:8800/"
GENERATE_ENDPOINT = f"{FASTAPI_URL}/generate"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RATE_LIMIT_DELAY = 1.0  # seconds between requests
SERVER_TIMEOUT = 5  # seconds to wait for server response
BATCH_SIZE = 20  # Single batch of 20 concurrent requests

def check_server_availability():
    """Check if the server is running and healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=SERVER_TIMEOUT)
        if response.status_code == 200:
            print("Server is running and healthy!")
            return True
        else:
            print(f"Server returned unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Could not connect to server. Please make sure the server is running at", FASTAPI_URL)
        return False
    except requests.exceptions.Timeout:
        print("Server connection timed out. Please check if the server is running.")
        return False
    except Exception as e:
        print(f"Error checking server availability: {str(e)}")
        return False

def create_test_payload():
    """Create a test payload with a more realistic medical image."""
    # Create a more realistic test image (224x224 pixels)
    # Generate random noise to simulate medical image texture
    img_array = np.random.normal(128, 30, (224, 224)).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Add some simulated features
    for _ in range(5):
        x = random.randint(0, 223)
        y = random.randint(0, 223)
        radius = random.randint(5, 15)
        value = random.randint(180, 255)
        for i in range(max(0, x-radius), min(224, x+radius)):
            for j in range(max(0, y-radius), min(224, y+radius)):
                if (i-x)**2 + (j-y)**2 <= radius**2:
                    img_array[j, i] = value
    
    img = Image.fromarray(img_array)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    return {
        "image": base64_image,
        "message": "Describe any abnormalities in this X-ray image.",
        "parameters": {
            "num_beams": 1,
            "temperature": 0.7,
            "max_new_tokens": 100,
            "max_length": 500
        }
    }

def send_request(payload, request_num, total_requests):
    """Send a single request to the FastAPI endpoint with retries."""
    start_time = time.time()
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            print(f"\nSending request {request_num}/{total_requests}...")
            response = requests.post(GENERATE_ENDPOINT, json=payload, timeout=300)
            end_time = time.time()
            
            if response.status_code == 200:
                duration = end_time - start_time
                print(f"Request {request_num} completed successfully in {duration:.2f} seconds")
                print(f"Response: {response.json()['response'][:100]}...")  # Print first 100 chars of response
                return duration
            else:
                print(f"Request {request_num} failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    wait_time = RETRY_DELAY * (retry_count + 1)
                    print(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                
        except requests.exceptions.RequestException as e:
            print(f"Request {request_num} failed: {str(e)}")
            retry_count += 1
            if retry_count < MAX_RETRIES:
                wait_time = RETRY_DELAY * (retry_count + 1)
                print(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                print(f"Request {request_num} failed after {MAX_RETRIES} retries")
                return None
        except Exception as e:
            print(f"Unexpected error in request {request_num}: {str(e)}")
            return None
    
    return None

def run_single_batch_test(batch_size=BATCH_SIZE):
    """Run a single batch of concurrent requests."""
    print(f"\n=== Starting Single Batch Test with {batch_size} Concurrent Requests ===")
    
    # Create unique payloads for each request
    payloads = [create_test_payload() for _ in range(batch_size)]
    
    # Process requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i, payload in enumerate(payloads):
            request_num = i + 1
            futures.append(executor.submit(send_request, payload, request_num, batch_size))
        
        # Collect results
        inference_times = []
        failed_requests = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                inference_times.append(result)
            else:
                failed_requests += 1
    
    return inference_times, failed_requests

def main():
    # First check if server is available
    if not check_server_availability():
        print("Exiting due to server unavailability.")
        sys.exit(1)
    
    # Test configuration
    batch_size = BATCH_SIZE  # 20 concurrent requests
    
    start_time = time.time()
    
    # Run the single batch test
    inference_times, failed_requests = run_single_batch_test(batch_size)
    
    # Calculate metrics
    total_time = time.time() - start_time
    inference_times = np.array(inference_times)
    
    if len(inference_times) > 0:
        median_time = np.median(inference_times)
        percentile_95 = np.percentile(inference_times, 95)
        percentile_99 = np.percentile(inference_times, 99)
        throughput = len(inference_times) / total_time
        
        print("\nPerformance Metrics:")
        print(f"Total requests completed: {len(inference_times)}/{batch_size}")
        print(f"Failed requests: {failed_requests}")
        print(f"Success rate: {(len(inference_times)/batch_size)*100:.2f}%")
        print(f"Median inference time: {1000*median_time:.4f} ms")
        print(f"95th percentile: {1000*percentile_95:.4f} ms")
        print(f"99th percentile: {1000*percentile_99:.4f} ms")
        print(f"Throughput: {throughput:.2f} requests/sec")
        print(f"Total test duration: {total_time:.2f} seconds")
    else:
        print("No successful requests were completed.")
        print(f"Failed requests: {failed_requests}")

if __name__ == "__main__":
    main()
