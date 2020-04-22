import sys
from pandas import Series,DataFrame,read_csv
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re
import csv

from pyspark import SparkContext, SparkConf
conf = SparkConf().setMaster("yarn-client").setAppName("julie_model").set("spark.storage.memoryFraction", "0.1")
sc = SparkContext(conf=conf, pyFiles=[
#sc = SparkContext("yarn-client", "julie_model", pyFiles=[
#sc = SparkContext("local[4]", "julie_model", pyFiles=[
    'country_run.py',
    'format_for_maps.py',
    'introduction.py',
    'kde.py',
    'lib_for_analysis.py',
    'lib_for_cc.py',
    'lib_for_country_run.py',
    'lib_for_data_reading.py',
    'lib_for_growth_model.py',
    'lib_for_prim.py',
    'perc.py'
])

def parse_params():
    include = []
    exclude = []
    params = sys.argv[1:]
    for p in params:
        if p[0] == '-':
            exclude.append(p[1:].upper())
        else:
            include.append(p.upper())
    return include, exclude

countries_include, countries_exclude = parse_params()
if countries_include:
    print("Including: ", countries_include)
elif countries_exclude:
    print("Excluding: ", countries_exclude)
else:
    print("Including all countries")

data2day      = 30
year          = 2030
ini_year      = 2007

nameofthisround = 'may2016'

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

climate_param_bounds     = read_csv(data+"climate_param_bounds.csv",index_col="var")
climate_param_thresholds = read_csv(data+"climate_param_thresholds.csv",index_col="var")

shares_stunting = read_csv(data+"share_stunting_who.csv")

#storage of results
baselines  = "{}/baselines_{}/".format(model,nameofthisround)
with_cc    = "{}/with_cc_{}/".format(model,nameofthisround)

# if not os.path.exists(baselines):
#     os.makedirs(baselines)
# if not os.path.exists(with_cc):
#     os.makedirs(with_cc)
	
#getting the list of available countries
list_csv=os.listdir(finalhhdataframes)
list_countries  = [re.search('(.*)_finalhhframe.csv', s).group(1) for s in list_csv]

all_surveys=dict()
for myfile in list_csv:
	cc = re.search('(.*)_finalhhframe.csv', myfile).group(1)
	all_surveys[cc]=read_csv(finalhhdataframes+myfile)

##############################################################################
# Convert all surveys to Spark broadcast variable - this means this large variable won't be repeated sent over the network
bc_all_surveys = sc.broadcast(all_surveys)
##############################################################################


#load codes, population
codes         = read_csv('wbccodes2014.csv')
codes_tables  = read_csv(ssp_folder+"ISO3166_and_R32.csv")
ssp_pop       = read_csv(ssp_folder+"SspDb_country_data_2013-06-12.csv",low_memory=False)
ssp_gdp       = read_csv(ssp_folder+"SspDb_compare_regions_2013-06-12.csv")
hhcat         = read_csv(scenar_folder+"hhcat_v2.csv")
industry_list = read_csv(scenar_folder+"list_industries.csv")

#drivers of scenarios
uncertainties    = ['shareag','sharemanu','shareemp','grserv','grag','grmanu','skillpserv','skillpag','skillpmanu','p','b']
lhssample        = read_csv(scenar_folder+"lhs-table-1500-11uncertainties.csv")
# lhssample = lhssample.loc[0,:]
ranges = DataFrame(columns=['min','max'],index=uncertainties)

impact_scenars = read_csv(scenar_folder+"impact-scenarios-low-high.csv")

#run scenarios
ssp_to_test   = [4,5]

datalist=(hhcat,industry_list,codes_tables,lhssample,ranges,ssp_pop,ssp_gdp,food_share_data,impact_scenars,price_increase_reg, malaria_bounds, diarrhea_bounds, shares_stunting, malaria_diff, diarrhea_shares, diarrhea_reg, climate_param_bounds, climate_param_thresholds,codes)

paramvar=(year,ini_year,data2day,ssp_to_test)

# countrycode = "MWI"
# scenar = 1

# Construct a long list of (country, scenario)
keys = []
#for countrycode in [cc for cc in all_surveys.keys() if cc not in ['IND','CHN','IDN','PAK','COL','AUT','BEL']]:
#for countrycode in [cc for cc in all_surveys.keys() if cc not in ['IND','CHN']]:
#for countrycode in ["BGR"]:
if countries_include:
    countries = [cc for cc in all_surveys.keys() if cc in countries_include]
elif countries_exclude:
    countries = [cc for cc in all_surveys.keys() if cc not in countries_exclude]
else:
    countries = all_surveys.keys()
print("Final country list: ", countries)

for countrycode in countries:
    for scenar in range(300):
        keys.append((countrycode, scenar))

# print("\n\n**KEYS**")
# print(keys)

# For a given (country, scenario) key, run the model and return the outputs
def run_a_key(key):
    import cvxopt.msk
    cvxopt.msk.env.putlicensepath('/home/wb478922/julie/poverty_model/mosek/mosek.lic')
    countrycode, scenar = key
    print("*"*70)
    print("CC + SCENAR")
    print(countrycode)
    print(scenar)
    outputs = country_run(countrycode,scenar,datalist,paramvar,bc_all_surveys.value)
    if outputs:
        forprim_bau,forprim_cc = country_run(countrycode,scenar,datalist,paramvar,bc_all_surveys.value)
        return (countrycode, scenar, forprim_bau, forprim_cc)
    else:
        return (countrycode, scenar, None, None)

##############################################################################
# These three lines are the only parts that run across the cluster
#NUM_PARTITIONS = 40 * 6 / 2                              # 40 cores x 6 machines, assume 2 cores for each run
NUM_PARTITIONS = len(keys) / 20
rdd_keys = sc.parallelize(keys, NUM_PARTITIONS)          # Split the keys list across the cluster
rdd_outputs = rdd_keys.map(run_a_key)                    # For each key, run the model, return a list of outputs corresponding to list of keys
outputs = rdd_outputs.collect()                          # Bring back this distributed list from the cluster to single machine
##############################################################################


fn_out_bau = "outputs_spark_bau.csv"
fn_out_cc = "outputs_spark_cc.csv"
countrycode, scenar, forprim_bau, forprim_cc = outputs[0]
if forprim_bau is not None and forprim_cc is not None:
    forprim_bau.to_csv(fn_out_bau, header=True)
    forprim_cc.to_csv(fn_out_cc, header=True)
for countrycode, scenar, forprim_bau, forprim_cc in outputs[1:]:
    if forprim_bau is not None and forprim_cc is not None:
        forprim_bau.to_csv(fn_out_bau, mode='a', header=False)
        forprim_cc.to_csv(fn_out_cc, mode='a', header=False)
