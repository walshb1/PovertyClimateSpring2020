from mainframe_class import mainframe
import pandas as pd

mf = mainframe('spring2020_BW')
print(mf.nameofthisround)

if True: mf.raw_to_skims(reset=False)

# mf.get_ini_year()