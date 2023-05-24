from PIL import Image
import os
#resize and convert (to jpg) all pictures in the planet folders and store in the pool folder

target_dir = "../data/pool/"
directory = "../data/planets/"
for plan in os.listdir(directory):
	for file in os.listdir(directory + plan):
		img_path = directory + plan + "/" + file

		img = Image.open(img_path)

		img_resized = img.resize((80, 80))

		img_resized.save(target_dir + file.split(".")[0] +".jpg")

