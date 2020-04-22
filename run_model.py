import sys
from pandas import Series,DataFrame,read_csv,set_option
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re

set_option('display.width', 220)
######################################
# Set parameters
#
nameofthisround = '2020spring'
#
data2day      = 365
year          = 2030
povline       = 1.90
#ini_year      = 2007; replace with get_ini_year() below


# These are for the Red Cross analysis
run_2050 = False; future_with_cc = False
if nameofthisround == 'RedX':
	run_2050 = True
	future_with_cc = True


######################################
# Set directories
#
# INPUT directories
model             = os.getcwd()
data              = model+'/data/'
scenar_folder     = model+'/scenar_def/'
finalhhdataframes = model+'/finalhhdataframes_GMD/' # intermediate files (1 per country), that Julie uses so she doesn't have to re-run all the time
ssp_folder        = model+'/ssp_data/'
iiasa_data        = model+'/iiasa_data/'
#data_gidd_csv     = model+'/data_gidd_csv_v4/' # full hhsurveys, I2D2 in my case...
#pik_data          = model+'/pik_data/'        # Julie says unnecessary
#
# OUTPUT directories
present    = "{}/{}_present/".format(model,nameofthisround)
baselines  = "{}/{}_baselines/".format(model,nameofthisround)
with_cc    = "{}/{}_with_cc/".format(model,nameofthisround)
baselines_2050  = "{}/{}_baselines_2050/".format(model,nameofthisround)
with_cc_2050    = "{}/{}_with_cc_2050/".format(model,nameofthisround)

for _ in [present,baselines,with_cc,baselines_2050,with_cc_2050]:
	if nameofthisround == 'test': os.rmdir(_)
	if not os.path.exists(_): os.makedirs(_)
	


######################################
# Load necessary files
#
#load climate change impacts data 
# ^ This will be IIASA inputs
price_increase_reg = read_excel(iiasa_data+"boundaries_food_FOLU.xlsx",sheet_name='boundaries_food_FOLU')
food_share_data = read_csv(data+"food_shares_wbreg.csv")
#
####
# Now WITH malaria!!!
malaria_diff    = read_csv(data+"malaria_diff_2006-2035.csv")
malaria_bounds  = read_csv(data+"malaria_bounds.csv",index_col="var")
diarrhea_bounds = read_csv(data+"diarrhea_bounds.csv",index_col="var")
#
#####
diarrhea_shares  = read_csv(data+"diarrhea_who.csv",index_col="Region")
diarrhea_shares.index=diarrhea_shares.index.str.strip()
diarrhea_reg    = read_csv(data+"wbccodes2014_who.csv",index_col="country")
#
#####
# No natural disasters!
fa                       = read_csv(data+"disasters/av_fa_for_cc_model.csv",index_col='iso3')
vulnerabilities          = read_csv(data+"disasters/vulnerabilities.csv",index_col='iso3')
#
#####
# Need to see what these are, as distinct from price_increase. Probably temperature?
climate_param_bounds     = read_csv(data+"climate_param_bounds.csv",index_col="var")
climate_param_thresholds = read_csv(data+"climate_param_thresholds.csv",index_col="var")
#
#
# No Stunting!
shares_stunting = None #read_csv(data+"share_stunting_who.csv")




######################################
# Get list of available countries...
list_csv=os.listdir(finalhhdataframes)
all_surveys=dict()
for myfile in [_ for _ in list_csv if 'DS_Store' not in _ and _ != '_Final_CPI_PPP_to_be_used.dta']:
	cc = re.search('(.*)_finalhhframe.csv', myfile).group(1)
	all_surveys[cc]=read_csv(finalhhdataframes+myfile)
#
####
#load codes, population
codes         = read_csv('wbccodes2014.csv')
codes_tables  = read_csv(ssp_folder+"ISO3166_and_R32.csv")
ssp_pop       = read_csv(ssp_folder+"SspDb_country_data_2013-06-12.csv",low_memory=False)
ssp_gdp       = read_csv(ssp_folder+"SspDb_compare_regions_2013-06-12.csv")
hhcat         = read_csv(scenar_folder+"hhcat_v2.csv")
industry_list = read_csv(scenar_folder+"list_industries.csv")
#
####
# Drivers of scenarios
uncertainties    = ['shareag','sharemanu','shareemp','grserv','grag','grmanu','skillpserv','skillpag','skillpmanu','p','b','voice']
lhssample        = read_csv(scenar_folder+"lhs-table-600-12uncertainties.csv")
ranges = DataFrame(columns=['min','max'],index=uncertainties)

impact_scenars = read_csv(scenar_folder+"impact-scenarios-low-high.csv").tail(1)
# ^ IIASA here? 




######################################
# Packaging for country_run()
#
datalist=(hhcat,industry_list,codes_tables,lhssample,ranges,ssp_pop,ssp_gdp,food_share_data,impact_scenars,price_increase_reg,malaria_bounds, 
	  diarrhea_bounds, shares_stunting, malaria_diff, diarrhea_shares, diarrhea_reg, climate_param_bounds, climate_param_thresholds,codes,fa,vulnerabilities)
#
ssp_to_test   = [2]
#

######################################
# Do the thing (TM)
_record_keeping = read_csv('results_summary/GMD_to_finalhhframe_status.csv').set_index('country')
for _c in ['BAU_complete','CC_complete']:
	if _c not in _record_keeping.columns: _record_keeping[_c] = None



##
for countrycode in list(all_surveys.keys()):
	if countrycode in ['ARG','LBN','FSM','MOZ','PSE','TUR','MKD']: continue 

	ini_year = get_ini_year(countrycode)
	
	paramvar=(year,ini_year,data2day,ssp_to_test,povline)
	scenar = 18
	
	if nameofthisround == 'RedX': switches = ['all','switch_ag_rev','switch_temp','switch_disasters']
	elif nameofthisround == 'results': switches = ['switch_ag_rev']
	elif nameofthisround == 'test': switches = []
	else: assert(False)

	if (not os.path.isfile("{}forprim_bau_{}.csv".format(baselines,countrycode)) 
	    or not os.path.isfile("{}forprim_cc_{}.csv".format(with_cc,countrycode))
	    or not os.path.isfile("{}forprim_now_{}.csv".format(present,countrycode))):
		print('\n\n',countrycode)
		
		b_value = 0
		forprim_now, forprim_bau,forprim_cc = country_run(countrycode,scenar,datalist,paramvar,all_surveys,switches,b_value,future_with_cc)

		if forprim_bau.shape[0]!=0 and forprim_cc.shape[0]!=0:
			if b_value == 0:
				forprim_now.T.to_csv("{}forprim_now_{}{}.csv".format(present,countrycode,('_redist' if b_value != 0 else '')))
			forprim_bau.T.to_csv("{}forprim_bau_{}{}.csv".format(baselines,countrycode,('_redist' if b_value != 0 else '')))
			forprim_cc.T.to_csv("{}forprim_cc_{}{}.csv".format(with_cc,countrycode,('_redist' if b_value != 0 else '')))

		_record_keeping.loc[countrycode,'BAU_complete'] = True if forprim_bau.shape[0]!=0 else False
		_record_keeping.loc[countrycode,'CC_complete'] = True if forprim_cc.shape[0]!=0 else False
	        #except: print('Did not run ',countrycode)


		if run_2050:  
			paramvar_2050=(2050,ini_year,data2day,ssp_to_test,povline)
			forprim_now, forprim_bau,forprim_cc = country_run(countrycode,scenar,datalist,paramvar_2050,all_surveys,switches)

			if forprim_bau.shape[0]!=0 and forprim_cc.shape[0]!=0:
				#forprim_now.T.to_csv("{}forprim_now_{}.csv".format(present,countrycode))
				forprim_bau.T.to_csv("{}forprim_bau_{}.csv".format(baselines_2050,countrycode))
				forprim_cc.T.to_csv("{}forprim_cc_{}.csv".format(with_cc_2050,countrycode))


	_record_keeping.sort_index().to_csv('results_summary/GMD_to_finalhhframe_status.csv')

