import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/medical-mnist")

print("Path to dataset files:", path)