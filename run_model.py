from create_dtaframes import mainframe
import pandas as pd

mf = mainframe('spring2020_BW')
print(mf.nameofthisround)

mf.raw_to_skims()
