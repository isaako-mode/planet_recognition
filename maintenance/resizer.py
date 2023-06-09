from PIL import Image
import os
#resize and convert (to jpg) all pictures in the planet folders and store in the pool folder

target_dir = "../data/pool/"
directory = "../data/planets/"
for plan in os.listdir(directory):
	i = 0
	for file in os.listdir(directory + plan):
		#print(file[-3:])

		img_path = directory + plan + "/" + file
		planet_prefix = plan[:3]

		img = Image.open(img_path)

		img_resized = img.resize((80, 80))

		try:

			img_resized.save(target_dir + planet_prefix + str(i) + ".png")
			i += 1
		except:
			continue

