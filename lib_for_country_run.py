import sys
from pandas import Series,DataFrame,read_csv, set_option
import os
from country_run import *
from lib_for_data_reading import *
from lib_for_cc import *
import re

set_option('display.width', 220)

def save_out_inputs(countrycode,scen,inputs,ini_pop_desc,future_pop_desc):

	if scen[:1] != '_': scen = '_'+scen
	
	_recB = read_csv('results_summary/scenario_definitions.csv').set_index('country')
	
	for _c in ['prod_growth_service'+scen,'prod_growth_agricul'+scen,'prod_growth_manufac'+scen,
		   'skilledworkers'+scen,'init_skilledworkers'+scen,
		   'urbanization'+scen,'employment'+scen]:
		if _c not in _recB.columns: _recB[_c] = None


	_recB.loc[countrycode,['prod_growth_service'+scen,'prod_growth_agricul'+scen,'prod_growth_manufac'+scen]] = [inputs['grserv'],inputs['grag'],inputs['grmanu']]
	_recB.loc[countrycode,['urbanization'+scen,'employment'+scen]] = [inputs['shareurban'],inputs['shareemp']]
	_recB.loc[countrycode,'init_skilledworkers'+scen] = float(ini_pop_desc['skillworkers']/(ini_pop_desc[['servworkers','agworkers','manuworkers']]).sum(axis=1))
	_recB.loc[countrycode,'skilledworkers'+scen] = float(future_pop_desc['skillworkers']/(future_pop_desc[['servworkers','agworkers','manuworkers']]).sum(axis=1))

	_recB.sort_index().to_csv('results_summary/scenario_definitions.csv')
	return True

def estimate_income_and_all(hhcat,hhdataframe,countrycode,year,with_person_data): 
	inc             = estime_income(hhcat,hhdataframe,countrycode,year,with_person_data)
	characteristics = keep_characteristics_to_reweight(hhdataframe)
	ini_pop_desc    = calc_pop_desc(characteristics,hhdataframe['weight'])
	inimin          = hhdataframe['Y'].min()
	return inc,characteristics,ini_pop_desc,inimin
	
def run_one_baseline(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,wbreg,data2day,food_share_data,istoobig=False):
	
	# What fraction of labor force in ag/manu/serv?
	# --> correct_shares just checks to make sure (ag+manu)<=1
	shareag,sharemanu = correct_shares(inputs['shareag'],inputs['sharemanu'])
	shareemp          = inputs['shareemp']
	shareurban        = inputs['shareurban']

	# so these 3 above are taken as inputs. 
	# --> Q. where do they come from?
	# --> A. inputs argument is one scenario from (one row of) scenarios defined from ranges in country_run
	# --> NB: I will need to change those.

	# This returns a description of the population (units = headcount), including labor participation & sectoral stats.
	future_pop_desc,pop_0014=build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp,shareurban,istransformationscen=False)

	# record growth rates for baseline:
	save_out_inputs(countrycode,'base(ssp='+str(ssp)+')',inputs,ini_pop_desc,future_pop_desc)

	# weights in future times. 
	weights_proj  = find_new_weights(characteristics,finalhhframe['weight'],future_pop_desc)
	weights_proj = DataFrame(weights_proj,index=finalhhframe.index,columns=["weight"])
	# ^ this is the use of mosek. the rest is just tweaks & measurements of the "new" population 

	futurehhframe              = futurehh(finalhhframe,pop_0014)
	futurehhframe['weight']    = weights_proj["weight"]

	income_proj,futureinc  = future_income_baseline(countrycode,inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,desc_str=('{}_{}'.format(ssp,year)))
	# in above function, I write sectoral growth out to 'results_sumary/sectoral_growth_rates_by_country.csv' 
	income_proj.fillna(0, inplace=True)
	futurehhframe['Y']         = income_proj

	return futurehhframe,futureinc

def run_one_cc_scenar(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,ccparam,wbreg,data2day,food_share_data,futurehhframe_bau,switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health):
	print(switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health)
	shareag,sharemanu = correct_shares(inputs['shareag'],inputs['sharemanu'])
	shareemp          = inputs['shareemp']
	shareurban        = inputs['shareurban']
	
	fprice_increase,farmer_rev_change,cyclones_increase,losses_poor,losses_rich,shares_outside,temp_impact,shockstunting,sh_people_stunt,th_nostunt,transmission,malaria_yr_occ,lostdays_mal,eventcost_mal,lostdays_dia,eventcost_dia,th_diarrhea,malaria_share,diarrhea_share,diarrhea_yr_occ,flood_share_poor,flood_share_nonpoor,drought_share,wind_share,surge_share = ccparam
	
	future_pop_desc,pop_0014=build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp,shareurban,istransformationscen=False)
	
	futurehhframe          = futurehhframe_bau
	if not switch_ag_rev:
		farmer_rev_change = 0
		
	if not switch_temp:
		temp_impact = 0
	
	income_proj,futureinc  = future_income_simple(countrycode,inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,transmission*farmer_rev_change,desc_str=('{}_{}'.format(ssp,year)))
	income_proj.fillna(0, inplace=True)
	futurehhframe['Y']     = income_proj
	
	if switch_ag_prices:
		futurehhframe = food_price_impact(futurehhframe,fprice_increase,wbreg,food_share_data,data2day)
	
	if switch_disasters:
		futurehhframe = shock(futurehhframe,wind_share,losses_poor,losses_rich)
		futurehhframe = shock(futurehhframe,surge_share,losses_poor,losses_rich)
		futurehhframe = shock_flood(futurehhframe,flood_share_poor,flood_share_nonpoor,losses_poor,losses_rich)
		futurehhframe = shock_drought(futurehhframe,drought_share,losses_poor)
	if switch_health:
		futurehhframe = stunting(futurehhframe,th_nostunt,wbreg,shockstunting,sh_people_stunt)
		futurehhframe = malaria_impact(futurehhframe,malaria_yr_occ,lostdays_mal,eventcost_mal,malaria_share)
		futurehhframe = diarrhea_impact(futurehhframe,diarrhea_yr_occ,lostdays_dia,eventcost_dia,diarrhea_share,th_diarrhea)
	
	return futurehhframe,futureinc

def run_transformation_scenar(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,ini_year,ccparam,wbreg,data2day,food_share_data,futurehhframe_bau,
			      switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health):
	print(switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health)

	shareag,sharemanu = correct_shares(inputs['shareag'],inputs['sharemanu'])
	shareemp          = inputs['shareemp']
	shareurban        = inputs['shareurban']
	
	fprice_increase,farmer_rev_change,cyclones_increase,losses_poor,losses_rich,shares_outside,temp_impact,shockstunting,sh_people_stunt,th_nostunt,transmission,malaria_yr_occ,lostdays_mal,eventcost_mal,lostdays_dia,eventcost_dia,th_diarrhea,malaria_share,diarrhea_share,diarrhea_yr_occ,flood_share_poor,flood_share_nonpoor,drought_share,wind_share,surge_share = ccparam
	# setting transmission (pass through) to 1, for max effect
	transmission = 1.0
	
	# Step 1: change future_pop_desc to include structural shifts to economy
	future_pop_desc,pop_0014=build_new_description(ini_pop_desc,ssp_pop,ssp,year,countrycode,shareag,sharemanu,shareemp,shareurban,istransformationscen=True)

	# record growth rates for transformation:
	save_out_inputs(countrycode,'tran (ssp='+str(ssp)+')',inputs,ini_pop_desc,future_pop_desc)

	# weights in future POST-TRANSFORMATION times. 
	weights_proj  = find_new_weights(characteristics,finalhhframe['weight'],future_pop_desc)
	weights_proj = DataFrame(weights_proj,index=finalhhframe.index,columns=["weight"])
	# ^ this is the use of mosek. the rest is just tweaks & measurements of the "new" population 

	futurehhframe              = futurehh(finalhhframe,pop_0014)
	futurehhframe['weight']    = weights_proj["weight"].clip(lower=0)
	
	income_proj,futureinc  = future_income_transformation(countrycode,inputs,year,finalhhframe,futurehhframe,inc,inimin,ini_year,shares_outside,temp_impact,transmission*farmer_rev_change,desc_str=('{}_{}'.format(ssp,year)))
	# in above function, I write sectoral growth out to 'results_sumary/sectoral_growth_rates_by_country.csv' 
	income_proj.fillna(0, inplace=True)
	futurehhframe['Y']     = income_proj
	
	futurehhframe = food_price_impact(futurehhframe,fprice_increase,wbreg,food_share_data,data2day)
	
	#if switch_disasters:
	#	futurehhframe = shock(futurehhframe,wind_share,losses_poor,losses_rich)
	#	futurehhframe = shock(futurehhframe,surge_share,losses_poor,losses_rich)
	#	futurehhframe = shock_flood(futurehhframe,flood_share_poor,flood_share_nonpoor,losses_poor,losses_rich)
	#	futurehhframe = shock_drought(futurehhframe,drought_share,losses_poor)
	#if switch_health:
	#	futurehhframe = stunting(futurehhframe,th_nostunt,wbreg,shockstunting,sh_people_stunt)
	#	futurehhframe = malaria_impact(futurehhframe,malaria_yr_occ,lostdays_mal,eventcost_mal,malaria_share)
	#	futurehhframe = diarrhea_impact(futurehhframe,diarrhea_yr_occ,lostdays_dia,eventcost_dia,diarrhea_share,th_diarrhea)
	
	return futurehhframe,futureinc
