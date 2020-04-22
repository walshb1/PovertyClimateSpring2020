Attached an excel spreadsheet giving you what you need (along with the shapefile with the countries, index numbers used are the same). It is on country basis, so you can aggregate in any way you like. 

The columns are named as follows:
Data_<ssp>_<rcp>_<GCM>_<period>_<return period number>

When present-day population is used, we simply use the term ‘base’. When present-day climate is used, we use ‘historical’.

Return period ‘5’ is 100-yr from our python list, i.e. the 5th in the list starting from pythonic index zero [2, 5, 10, 25, 50, 100, 250, 500, 1000].

I think you need to average columns E, G, I, K, M to get 2080 averages and F, H, J, L, N for 2030 averages.
