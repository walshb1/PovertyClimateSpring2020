import sys
from pandas import Series,DataFrame,read_csv
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re

data2day      = 30
year          = 2030
ini_year      = 2007
povline       = 1.25

nameofthisround = 'may2017'

model             = os.getcwd()
data              = model+'/data/'
scenar_folder     = model+'/scenar_def/'
finalhhdataframes = model+'/finalhhdataframes/'
ssp_folder        = model+'/ssp_data/'
pik_data          = model+'/pik_data/'
iiasa_data        = model+'/iiasa_data/'
data_gidd_csv     = model+'/data_gidd_csv_v4/'

#load climate change impacts data
price_increase_reg = read_csv(iiasa_data+"boundaries_food2.csv")
food_share_data = read_csv(data+"food_shares_wbreg.csv")

malaria_diff    = read_csv(data+"malaria_diff_2006-2035.csv")
malaria_bounds  = read_csv(data+"malaria_bounds.csv",index_col="var")
diarrhea_bounds = read_csv(data+"diarrhea_bounds.csv",index_col="var")

diarrhea_shares  = read_csv(data+"diarrhea_who.csv",index_col="Region")
diarrhea_shares.index=diarrhea_shares.index.str.strip()
diarrhea_reg    = read_csv(data+"wbccodes2014_who.csv",index_col="country")

fa                       = read_csv(data+"disasters/av_fa_for_cc_model.csv",index_col='iso3')
vulnerabilities          = read_csv(data+"disasters/vulnerabilities.csv",index_col='iso3')

climate_param_bounds     = read_csv(data+"climate_param_bounds.csv",index_col="var")
climate_param_thresholds = read_csv(data+"climate_param_thresholds.csv",index_col="var")

shares_stunting = read_csv(data+"share_stunting_who.csv")

#storage of results
baselines  = "{}/baselines_{}/".format(model,nameofthisround)
with_cc    = "{}/with_cc_{}/".format(model,nameofthisround)

if not os.path.exists(baselines):
	os.makedirs(baselines)
if not os.path.exists(with_cc):
	os.makedirs(with_cc)
	
#getting the list of available countries
list_csv=os.listdir(finalhhdataframes)
all_surveys=dict()
for myfile in list_csv:
	cc = re.search('(.*)_finalhhframe.csv', myfile).group(1)
	all_surveys[cc]=read_csv(finalhhdataframes+myfile)

#load codes, population
codes         = read_csv('wbccodes2014.csv')
codes_tables  = read_csv(ssp_folder+"ISO3166_and_R32.csv")
ssp_pop       = read_csv(ssp_folder+"SspDb_country_data_2013-06-12.csv",low_memory=False)
ssp_gdp       = read_csv(ssp_folder+"SspDb_compare_regions_2013-06-12.csv")
hhcat         = read_csv(scenar_folder+"hhcat_v2.csv")
industry_list = read_csv(scenar_folder+"list_industries.csv")

#drivers of scenarios
uncertainties    = ['shareag','sharemanu','shareemp','grserv','grag','grmanu','skillpserv','skillpag','skillpmanu','p','b','voice']
lhssample        = read_csv(scenar_folder+"lhs-table-600-12uncertainties.csv")
ranges = DataFrame(columns=['min','max'],index=uncertainties)

impact_scenars = read_csv(scenar_folder+"impact-scenarios-low-high.csv")

#run scenarios
datalist=(hhcat,industry_list,codes_tables,lhssample,ranges,ssp_pop,ssp_gdp,food_share_data,impact_scenars,price_increase_reg, malaria_bounds, diarrhea_bounds, shares_stunting, malaria_diff, diarrhea_shares, diarrhea_reg, climate_param_bounds, climate_param_thresholds,codes,fa,vulnerabilities)

switches = ['all']

for ssp_to_test in [4,5]:

	chosen_scenarios = read_csv("ssp{}_scenarios.csv".format(ssp_to_test))

	paramvar=(year,ini_year,data2day,[ssp_to_test],povline)
	
	for ii in chosen_scenarios.index:
		countrycode = reverse_correct_countrycode(chosen_scenarios.loc[ii,'countrycode'])
		scenar = chosen_scenarios.loc[ii,'scenar']
		forprim_bau,forprim_cc = country_run(countrycode,scenar,datalist,paramvar,all_surveys,switches)

			
