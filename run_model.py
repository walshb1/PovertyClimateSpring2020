from mainframe_class import mainframe
import pandas as pd

mf = mainframe('spring2020_BW')
print(mf.nameofthisround)

if False: mf.raw_to_skims(reset=False)

for i in mf.country_file_dict:
	print(mf.country_file_dict[i])