import os
from locust import HttpUser, task, between
from random import choice


class QuickstartUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        self.client.headers = {"X-Api-Key": "API_KEY"}
        # Get the list of image files in the "happy" folder
        self.image_files = [f for f in os.listdir(
            "happy") if os.path.isfile(os.path.join("happy", f))]

    @task
    def predictImages(self):
        # Randomly select an image from the folder
        image_path = os.path.join("happy", choice(self.image_files))

        # Prepare the files dictionary for the POST request
        files = {'file': (image_path, open(image_path, 'rb'), 'image/jpg')}

        # Make a POST request to the predict_images endpoint
        response = self.client.post(
            "/predict", files=files)

        # Validate the response
        if response.status_code != 200:
            print(
                f"Failed to predict image. Status Code: {response.status_code}")
            return

        # Assuming the response contains a JSON with an 'emotion' field
        result = response.json()
        emotion = result.get('emotion', 'Unknown')
        print(f"Predicted emotion: {emotion}")

    @task
    def predictMultipleImages(self):
        # Select multiple images from the folder
        selected_images = [os.path.join(
            "happy", choice(self.image_files)) for _ in range(10)]

        # Prepare the files dictionary for the POST request with multiple images
        files = [('files', (image_path, open(image_path, 'rb'), 'image/jpg'))
                 for image_path in selected_images]

        # Make a POST request to the predict_images endpoint with multiple images
        self.client.post("/process_images", files=files)
