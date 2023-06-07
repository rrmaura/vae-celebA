import os
import shutil

# Set the paths for the original folder and the new subfolder
original_folder = r"C:\Users\Roberto\Desktop\celebA\img_align_celeba\img_align_celeba"
new_subfolder = r"C:\Users\Roberto\Desktop\celebA\img_align_celeba\test_set"


# Create the new subfolder if it doesn't exist
if not os.path.exists(new_subfolder):
    os.makedirs(new_subfolder)

# Get the list of all files in the original folder
all_files = os.listdir(original_folder)

# Iterate through each file
for file_name in all_files:
    # Extract the image number from the file name
    image_number = int(file_name.split(".")[0])

    # Check if the image number falls within the test set range
    if 182638 <= image_number <= 202599:
        # Create the path for the original file
        original_file_path = os.path.join(original_folder, file_name)

        # Create the path for the new file in the subfolder
        new_file_path = os.path.join(new_subfolder, file_name)

        # Copy the file to the subfolder
        shutil.copyfile(original_file_path, new_file_path)
