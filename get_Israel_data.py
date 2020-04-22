from pandas import read_excel,concat,Series,read_stata,DataFrame,read_csv,isnull,notnull
import os

model=os.getcwd()
data_gidd=model+'/data_gmd_dta_july2016/'
data_gidd_csv=model+'/data_gmd_csv_july2016/'

if not os.path.exists(data_gidd_csv):
	os.makedirs(data_gidd_csv)

def change_extension(mystring):
	"replaces dta by csv in a string"
	s=mystring.split('.')
	s.remove('dta')
	s.append('csv')
	return ".".join(s)
		
for myfile in os.listdir(data_gidd):
	path = os.path.join(data_gidd, myfile)
	if os.path.isdir(path):
		# skip directories
		continue
	# if ('NER' in myfile)|('STP' in myfile)|('TCD' in myfile):
	if ('.dropbox' in myfile)|('NER' in myfile)|('COG' in myfile)|('CPV' in myfile)|('GIN' in myfile)|('MRT' in myfile)|('STP' in myfile)|('TCD' in myfile)|('SYC' in myfile):
		continue
	if not os.path.isfile(data_gidd_csv+change_extension(myfile)):
		print(myfile)
		dt=read_stata(data_gidd+myfile,encoding='utf8')
		dt.to_csv(data_gidd_csv+change_extension(myfile),encoding='utf8')