from pandas import concat,Series,DataFrame,read_csv,HDFStore,set_option
from lib_for_growth_model import *
from lib_for_cc import *
from lib_for_country_run import *
import numpy as np
import os

switches_as_cols = ['switch_ag_rev','switch_temp','switch_ag_prices','switch_disasters','switch_health']

def country_run(countrycode,scenar,datalist,paramvar,all_surveys,switches,b_value=0,future_with_cc=False, with_person_data=True):
	
	# Runs one scenario for one country (with 1 SSP(2+4 for now) and baseline/climate)
	# --> Returns a dataframe with aggregated results per scenario.
	
	# Inputs: 
	# --> countrycode: an ISO3 code
	# --> scenar: number of the scenario to run
	# --> datalist: all data required to run the scenario
	# --> paramvar: all parameters defined in run_model.py
	# --> all_surveys: dict of dataframes, indexed by countrycode, with household surveys for all countries
	
	##################
	# STEP 1: Split up the inputs
	(year,ini_year,data2day,ssp_to_test,povline) = paramvar
	#
	(hhcat,industry_list,codes_tables,lhssample,ranges,ssp_pop,ssp_gdp,food_share_data,impact_scenars,price_increase_reg,malaria_bounds, diarrhea_bounds, 
	 shares_stunting, malaria_diff, diarrhea_shares, diarrhea_reg, climate_param_bounds,climate_param_thresholds,codes,fa,vulnerabilities) = datalist
	

	##################
	# STEP 2: Get the country data from list of surveys
	out = filter_country(countrycode,all_surveys,codes)
	if out is None:	
		_ = DataFrame([[countrycode,year,scenar]],columns=['country','year','scenar'])
		return _,_,_
	else: finalhhframe,countrycode,istoobig,wbreg = out
	print(np.min(finalhhframe.weight*finalhhframe.nbpeople), np.min(finalhhframe.weight), np.min(finalhhframe.nbpeople))
	finalhhframe.dropna(inplace=True)
	finalhhframe.reset_index(inplace=True, drop=True)
	print('Step 2 complete')
	
	
	##################
	# STEP 3: estimate income
	print(ini_year,'to',year)

	inc,characteristics,ini_pop_desc,inimin = estimate_income_and_all(hhcat,finalhhframe,countrycode,ini_year,with_person_data=with_person_data)
	print('Step 3 complete')
	#
	# characteristics = dataframe with row per hh, #individuals who are 
	# -------------->   {old, children, unemployed, skilled, urban, agworkers, manuworkers, servworkers} 
	#
	# ini_pop_desc    = series with weighted sums of individuals, for all columns in characteristics 
	#
	# inimin          = hhdataframe['Y'].min() 

	##################
	# STEP 4: initialize outputs
	forprim_bau = DataFrame()
	forprim_cc  = DataFrame()
	forprim_now  = DataFrame()	

	##################
	# STEP 5: get the ranges of values to generate the scenarios from an LHS (latin-hypercube sampling) table
	ranges,ini_pop_out    = scenar_ranges(ssp_to_test[0],ranges,finalhhframe,countrycode,ssp_gdp,codes_tables,ssp_pop,year,ini_year,b_value)
	scenarios = lhssample.values*np.diff(ranges[['min','max']].values).T+ranges['min'].values
	scenarios = DataFrame(scenarios,columns=ranges.index)
	print('Step 5 complete')
	
	# Note: in the version designed for the cluster, we get run only one scenario (called scenar) per call of this function
	inputs = scenarios.ix[scenar,:]
	# ^ insert loop over rows in scenarios here
	#inputs = scenarios.mean()

	print('\n\nRunning mosek fit with these input values:\n',inputs.head(15))
	

	##################
	# STEP 6: Split the survey (if necessary)
	# --> it takes a lot of time so we don't repeat it in each ssp scenario
	# --> the big df is split by regions and each subdataframe is reduced by merging households
	if istoobig:
		finalhhframes = split_big_dframe(finalhhframe,hhcat)
		finalhhframe1,finalhhframe2 = finalhhframes
		finalhhframe  = concat([finalhhframe1,finalhhframe2],axis=0)
	else: finalhhframes = finalhhframe
	
	
	##################
	# STEP 6.5: save out now results
	finalhhframes = finalhhframes.fillna(0)
	pop_of_interest = (finalhhframes.Y<(1.90*365)).to_frame(name='190')
	pop_of_interest['320'] = (finalhhframes.Y<(3.20*365))
	pop_of_interest['1000'] = (finalhhframes.Y<(10.00*365))
	pop_of_interest.index.name = 'hhid'
	pop_of_interest.to_csv('exposed_pop/{}.csv'.format(countrycode))

	indicators_now  = calc_indic(countrycode,
				     finalhhframes['Y'],
				     finalhhframes['weight']*finalhhframes['nbpeople'],
				     finalhhframes['weight'],
				     finalhhframes,
				     data2day,finalhhframes['Y'],povline,ini_year)

	ini_pop_out = ini_pop_out.T.squeeze()
	forprim_now = forprim_now.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp_to_test[0])]],columns=['country','year','scenar','ssp']),
						 indicators_now,
						 DataFrame([ini_pop_out.values],columns=ini_pop_out.index).drop('unemployed',axis=1)],axis=1),ignore_index=True)
	print('Step 6.5 complete')


	##################
	# STEP 7: run_one_baseline() function	
	for ssp in ssp_to_test:

		if ssp != ssp_to_test[0]:
                        ################
			# STEP 5: get the ranges of values to generate the scenarios from an LHS (latin-hypercube sampling) table
			ranges,ini_pop_out    = scenar_ranges(ssp,ranges,finalhhframe,countrycode,ssp_gdp,codes_tables,ssp_pop,year,ini_year,b_value)
			scenarios = lhssample.values*np.diff(ranges[['min','max']].values).T+ranges['min'].values
			scenarios = DataFrame(scenarios,columns=ranges.index)
			inputs = scenarios.ix[scenar,:]
			#inputs = scenarios.mean()

		
		# STEP 7A: run the baseline scenario
		# --> run_one_baseline() function
		print("I am running baseline {} with ssp {} of {}".format(scenar,ssp,countrycode))
		futurehhframe_bau,futureinc_bau = run_one_baseline(ssp,inputs,finalhhframes,ini_pop_desc,ssp_pop,year,countrycode,characteristics,
								   inc,inimin,ini_year,wbreg,data2day,food_share_data,istoobig)


		if sum(futurehhframe_bau['weight'])==0:
			futurehhframe_bau = futurehhframe_bau.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp)]],columns=['country','year','scenar','ssp']),DataFrame([inputs.values],columns=inputs.index)],ignore_index=True))
			continue
		
		# STEP 7B: Pull out results
		# --> calc_indic() function
		indicators_bau  = calc_indic(countrycode,
					     futurehhframe_bau['Y'],
					     futurehhframe_bau['weight']*futurehhframe_bau['nbpeople'],
					     futurehhframe_bau['weight'],
					     futurehhframe_bau,
					     data2day,futureinc_bau,povline)
		
		# STEP 7C: additional results: 
                # --> implied average productivity growth for all workers (skilled and unskilled) by sector
		# --> actual_productivity_growth() function
		prod_gr_serv_bau,prod_gr_ag_bau,prod_gr_manu_bau = actual_productivity_growth(futurehhframe_bau,inc,futurehhframe_bau,futureinc_bau,year,ini_year)
		
		# STEP 7D: Store results
		forprim_bau = forprim_bau.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp)]],columns=['country','year','scenar','ssp']),
							 indicators_bau,
							 DataFrame([inputs.values],columns=inputs.index),
							 DataFrame([[prod_gr_serv_bau,prod_gr_ag_bau,prod_gr_manu_bau]],columns=['prod_gr_ag','prod_gr_serv','prod_gr_manu'])
							 ],axis=1),
						 ignore_index=True)
		
		for ccint in impact_scenars.index:			
			for switch in switches:
				if switch == 'all':
					switch_temp      = True
					switch_disasters = True
					switch_ag_prices = True
					switch_ag_rev    = True
					switch_health    = True
				else:
					switch_ag_rev    = switch == 'switch_ag_rev'
					switch_temp      = switch == 'switch_temp'
					switch_ag_prices = switch == 'switch_ag_prices'
					switch_disasters = switch == 'switch_disasters'
					switch_health    = switch == 'switch_health'
								
				# --> runs the climate impacts scenarios. Here no need to recalculate the weights
				# STEP 7Di: extracts cc parameters
				food_cursor, temp_impact, paramshstunt, malaria_cursor, diarrhea_cursor, disasters_cursor = impact_scenars.loc[ccint,:]
				ccparam,ccparam2keep,ccparam2keeptitles = get_model_cc_parameters(food_cursor, temp_impact, paramshstunt, malaria_cursor, diarrhea_cursor, disasters_cursor, 
												  ssp, wbreg, price_increase_reg, malaria_bounds, diarrhea_bounds, finalhhframe, 
												  shares_stunting, malaria_diff, countrycode, diarrhea_shares, diarrhea_reg, 
												  climate_param_bounds, climate_param_thresholds, fa,vulnerabilities,inputs.voice)

				# STEP 7Dii: run the cc scenario
				print("I am running climate scenario {} of baseline {} with ssp {} of {}".format(ccint,scenar,ssp,countrycode))
				if not future_with_cc:
					futurehhframe_cc,futureinc_cc = run_transformation_scenar(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,
												  ini_year,ccparam,wbreg,data2day,food_share_data,futurehhframe_bau,
												  switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health)
				else:
					futurehhframe_cc,futureinc_cc = run_one_cc_scenar(ssp,inputs,finalhhframe,ini_pop_desc,ssp_pop,year,countrycode,characteristics,inc,inimin,
											  ini_year,ccparam,wbreg,data2day,food_share_data,futurehhframe_bau,
											  switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health)
					
				if sum(futurehhframe_cc['weight'])==0:
					futurehhframe_cc = futurehhframe_cc.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp)]],columns=['country','year','scenar','ssp']),
											   DataFrame([inputs.values],columns=inputs.index)],ignore_index=True))
					continue
	
				# STEP 7Diii: calculate indicators and store results
				indicators_cc  = calc_indic(countrycode,
							    futurehhframe_cc['Y'],
							    futurehhframe_cc['weight']*futurehhframe_cc['nbpeople'],
							    futurehhframe_cc['weight'],
							    futurehhframe_cc,
							    data2day,futureinc_cc,povline)

				prod_gr_serv_cc, prod_gr_ag_cc, prod_gr_manu_cc = actual_productivity_growth(futurehhframe_cc, inc,futurehhframe_cc, futureinc_cc, year,ini_year)
				switch_cols = ['switch_ag_rev','switch_temp','switch_ag_prices','switch_disasters','switch_health']
				switches_values = DataFrame([[switch_ag_rev,switch_temp,switch_ag_prices,switch_disasters,switch_health]],columns=switch_cols)
							
				forprim_cc  = forprim_cc.append(concat([DataFrame([[countrycode,year,scenar,'ssp{}'.format(ssp),ccint]],columns=['country','year','scenar','ssp','ccint']),
									switches_values,
									indicators_cc,
									DataFrame([inputs.values],columns=inputs.index),
									DataFrame([ccparam2keep],columns=ccparam2keeptitles),
									DataFrame([[prod_gr_serv_cc,prod_gr_ag_cc,prod_gr_manu_cc]],columns=['prod_gr_ag','prod_gr_serv','prod_gr_manu'])],
								       axis=1),ignore_index=True)
	
	return forprim_now,forprim_bau,forprim_cc, inc

