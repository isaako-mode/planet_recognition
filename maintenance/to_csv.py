from PIL import Image
import numpy as np
import os
import csv
import pandas as pd

planet_vals = {"mer" : 0, "ven" : 1, "ear" : 2, "mar" : 3, "jup" : 4, "sat" : 5, "ura" : 6, "nep" : 7}

data = {"planet" : [], "pixels" : []}

df = pd.DataFrame.from_dict(data)

directory = "../data/pool/"
file_name = "../data/data.csv"

pixels = []


for file in os.listdir(directory):
	img = Image.open(directory + file)

	#Convert image to NumPy array
	arr = np.asarray(img)

	for ls in arr[2]:
		for pixel in ls:
			pixels.append(int(pixel))

	prefix = file[:3]

	df = df._append({"planet" : int(planet_vals[prefix]), "pixels" : pixels}, ignore_index = True)

	pixels = []

print(df)


df.to_csv(file_name)

