import pandas as pd
import matplotlib
import numpy as np

from datetime import datetime
import mb_query

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

def h10_xray(filepath_of_csv):
    df = pd.read_csv(filepath_of_csv)
    #changes the columns from H10 into strings where needed - I don't like objects
    for x in ['Product Details','ASIN','Brand','Category','Size Tier','Fulfillment','Dimensions','Weight','Images']:
        df[x] = df[x].astype('string')
    #makes the date of asin create an actual dtype    
    for x in ['Creation Date']:
        df[x] = pd.to_datetime(df[x], infer_datetime_format=True)
    #makes the numbers into floats rather than objects which suck    
    for x in ['Sales','Revenue','BSR','Review Count','Review velocity','Weight','Images']:
        try:
            df[x] = df[x].fillna(0)
        except: 
            pass
        try:
            df[x] = df[x].str.replace(',', '').astype(float)
        except:
            pass
    #sorts by sales, not sure if needed
    df.sort_values(by=['Sales'],ascending=[False],inplace=True)
    #removes sponsored products (turns out ~ means "not in")
    df = df[~df['Product Details'].str.contains('($) ',regex=False)]
    return df

def seasonality(s_curve):
    #builds a seasonality curve by month for the data we have. s_curve is a user defined variable later, haven't really considered discount in here yet, will cover later, not sure it makes a huge difference to total forecast
    d = {'Month':months(),'Demand %':s_curve,'Price Discount':price_discount_curve}
    x = pd.DataFrame(data=d)
    return x

def months():
    #literally a list of months, why this is in a function i don't know, but kind of OOP!
    x = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return x

def exponenter(count):
    #works out the curve of search results based on ranking
    if count > 50:
        exponent = -2.5
    elif count > 40:
        exponent = -2
    elif count >30:
        exponent = -1.5
    elif count >20:
        exponent= -1
    elif count > 10:
        exponent = -0.9
    elif count >5:
        exponent = -0.8
    elif count >3:
        exponent = -0.7
    elif count >=1:
        exponent = -0.6
    return exponent

def full_year_generation(df_in):
    #pull the seasonality curve into a df
    szn = seasonality(s_curve)
    #make some months
    d = {'Month':months()}
    #blank dataframe just including months
    fy = pd.DataFrame(data = d)
    #normalizes the data from the seasonality curve supplied (i.e. if the data is from a peak, it smooths down the full year total based on the % of demand for the "forecast month" in the seasonality curve)
    current_month = datetime.now().month
    if current_month -1 == 0:
        forecast_month = 12
    else:
        forecast_month = current_month-1
    #forecast month here is -1 because of python 0 indexing (the python list starts from 0 and my list starts from 0)
    forecast_month_percent_of_demand = s_curve[forecast_month-1]
    #loop through the imported dataframe to find the sum of each of them and then multiply by the s_curve (again user defined)
    for x in ['Sales','Revenue']:
        #normalizes to day (using monthly seasonality curve) then brings up to year, think this is better than dividing by 12 because H10 data is T30D only
        y = ((df_in[x].sum())/forecast_month_percent_of_demand)
        fy[x]= szn['Demand %']*y
    #sales is a dumb and confusing name for a column which refers to units, so fixing that
    fy.rename(columns={'Sales':'Units'},inplace=True)
    #calc asp and pull other things from global variables
    #asp is calculated including the revenue discount
    fy['ASP'] = (fy['Revenue']/(1+szn['Price Discount']))/fy['Units']
    #we then correct units up assuming a price elasticity of 1 which keeps revenue flat (rough assumption - need to make sure we have a proper elasticity calculation at some point)
    fy['Units'] = fy['Units']*(1+szn['Price Discount'])
    #then we recalculate revenue as a sense check
    fy['Revenue'] = fy['Units']*fy['ASP']
    fy['Conversion'] = conversion_percent
    #glance views are the number of views a product gets, conversion is a rough assumption, we'll use the GVs later to generate some kind of traffic analysis
    fy['GVs'] = fy['Units']/fy['Conversion']
    fy['Organic Search GVs'] = fy['GVs']*organic_search_percent
    fy['SSPA GVs'] = fy['GVs']*sspa_percent
    fy['Other GVs'] = fy['GVs']*other_gv_percent
    #an attempt to transpose the data, though I'm not really sure it needs to be done here so have commented it out
    #fy = fy.set_index('Month')
    #fy = fy.T.reset_index().rename(columns={'index':'Attributes'})
    return fy

#we do the same function as above but for the top ASIN by sales identified in the dataframe as a sense check for later   
def top_asin_full_year(df_in):
    #before i forget - iloc[0] pulls the first row and applies the same logic to it as above (following same seasonality curve)
    df_in = df_in.iloc[0]
    x = full_year_generation(df_in)
    return x

def cost_calculation(shipment_model,container):
    product_volume_cube = product_width*product_height*product_depth
    #fixed by master cube size
    carton_height = 0.6
    carton_width = 0.4
    carton_depth = 0.4
    carton_cube = carton_height*carton_depth*carton_width
    #fixed by container size
    twenty_foot_cbm = 25
    forty_foot_cbm = 55
    forty_foot_high_cube_cbm = 65
    #added 15% on top to deal with mastercube size
    product_cbm = product_volume_cube*1.15
    #volume calculations
    units_in_cbm = 1/product_cbm
    units_in_carton = carton_cube/product_cbm
    #SEA FREIGHT - make the dfs based on user input of container prices
    twenty_foot = pd.DataFrame(data = {'Name':['Bull','Base','Bear'],'US':twenty_foot_us,'UK':twenty_foot_uk,'EU':twenty_foot_eu})
    forty_foot = pd.DataFrame(data = {'Name':['Bull','Base','Bear'],'US':forty_foot_us,'UK':forty_foot_uk,'EU':forty_foot_eu})
    forty_foot_high_cube = pd.DataFrame(data = {'Name':['Bull','Base','Bear'],'US':forty_foot_high_cube_us,'UK':forty_foot_high_cube_uk,'EU':forty_foot_high_cube_eu})
    #add calculations per line item in DF - please ignore how awful this is - I couldn't work out how to do it in a loop
    twenty_foot['$ per CBM - US'] = twenty_foot['US']/twenty_foot_cbm
    twenty_foot['Freight per Unit - US'] = twenty_foot['$ per CBM - US']/units_in_cbm
    twenty_foot['$ per CBM - UK'] = twenty_foot['UK']/twenty_foot_cbm
    twenty_foot['Freight per Unit - UK'] = twenty_foot['$ per CBM - UK']/units_in_cbm
    twenty_foot['$ per CBM - EU'] = twenty_foot['EU']/twenty_foot_cbm
    twenty_foot['Freight per Unit - EU'] = twenty_foot['$ per CBM - EU']/units_in_cbm
    forty_foot['$ per CBM - US'] = forty_foot['US']/forty_foot_cbm
    forty_foot['Freight per Unit - US'] = forty_foot['$ per CBM - US']/units_in_cbm
    forty_foot['$ per CBM - UK'] = forty_foot['UK']/forty_foot_cbm
    forty_foot['Freight per Unit - UK'] = forty_foot['$ per CBM - UK']/units_in_cbm
    forty_foot['$ per CBM - EU'] = forty_foot['EU']/forty_foot_cbm
    forty_foot['Freight per Unit - EU'] = forty_foot['$ per CBM - EU']/units_in_cbm
    forty_foot_high_cube['$ per CBM - US'] = forty_foot_high_cube['US']/forty_foot_high_cube_cbm
    forty_foot_high_cube['Freight per Unit - US'] = forty_foot_high_cube['$ per CBM - US']/units_in_cbm
    forty_foot_high_cube['$ per CBM - UK'] = forty_foot_high_cube['UK']/forty_foot_high_cube_cbm
    forty_foot_high_cube['Freight per Unit - UK'] = forty_foot_high_cube['$ per CBM - UK']/units_in_cbm
    forty_foot_high_cube['$ per CBM - EU'] = forty_foot_high_cube['EU']/forty_foot_high_cube_cbm
    forty_foot_high_cube['Freight per Unit - EU'] = forty_foot_high_cube['$ per CBM - EU']/units_in_cbm
    #AIR FREIGHT no idea why the volumetric weight is *167 - taken from Rish's sheet, so probably is right?
    air_freight = pd.DataFrame(data={'Name':['Bull','Base','Bear'],'US':air_us,'UK':air_uk,'EU':air_eu})
    volumetric_weight = product_cbm*167
    if unit_weight>volumetric_weight:
        air_weight = unit_weight
    else:
        air_weight = volumetric_weight
    #work out airfreight costs to each locale    
    air_freight['Freight per Unit - US'] = air_freight['US']*air_weight
    air_freight['Freight per Unit - UK'] = air_freight['UK']*air_weight
    air_freight['Freight per Unit - EU'] = air_freight['EU']*air_weight
    #IMPORT DUTY
    import_us = product_cost_usd*import_duty_us
    import_eu = product_cost_usd*import_duty_eu
    import_uk = product_cost_usd*import_duty_uk
    #FBA Prep
    carton_distribution_cost_us = 3.40
    carton_distribution_cost_uk = 2.00
    carton_distribution_cost_eu = 1.20
    unit_prep_us = carton_distribution_cost_us/units_in_carton
    unit_prep_uk = carton_distribution_cost_uk/units_in_carton
    unit_prep_eu = carton_distribution_cost_eu/units_in_carton
    #FBA Delivery
    carton_cost_to_fba_us = 4.00
    carton_cost_to_fba_uk = 4.00
    carton_cost_to_fba_eu = 4.00
    unit_distribution_us = carton_cost_to_fba_us/units_in_carton
    unit_distribution_uk = carton_cost_to_fba_uk/units_in_carton
    unit_distribution_eu = carton_cost_to_fba_eu/units_in_carton
    #cost of captial
    cost_of_capital = 0.25
    #pick shipment cost based on input
    if container == '20':
        if shipment_model == 'Bull':
            sea = twenty_foot.loc[twenty_foot['Name'] == 'Bull'].copy()
        elif shipment_model == 'Base':
            sea = twenty_foot.loc[twenty_foot['Name'] == 'Base'].copy()
        elif shipment_model == 'Bear':
            sea = twenty_foot.loc[twenty_foot['Name'] == 'Bear'].copy()
    elif container == '40':
        if shipment_model == 'Bull':
            sea = forty_foot[forty_foot['Name'] == 'Bull'].copy()
        elif shipment_model == 'Base':
            sea = forty_foot[forty_foot['Name'] == 'Base'].copy()
        elif shipment_model == 'Bear':
            sea = forty_foot[forty_foot['Name'] == 'Bear'].copy()
    elif container == '40 High Cube':
        if shipment_model == 'Bull':
            sea = forty_foot_high_cube[forty_foot_high_cube['Name'] == 'Bull'].copy()
        elif shipment_model == 'Base':
            sea = forty_foot_high_cube[forty_foot_high_cube['Name'] == 'Base'].copy()
        elif shipment_model == 'Bear':
            sea = forty_foot_high_cube[forty_foot_high_cube['Name'] == 'Bear'].copy()
    if shipment_model == 'Bull':
        air = air_freight.loc[air_freight['Name']=='Bull'].copy()
    elif shipment_model == 'Base':
        air = air_freight.loc[air_freight['Name']=='Base'].copy()
    elif shipment_model == 'Bear':
        air = air_freight.loc[air_freight['Name']=='Bear'].copy()
    #add duties
    sea['Import Duty - US'] = import_us
    sea['Import Duty - UK'] = import_uk
    sea['Import Duty - EU'] = import_eu
    air['Import Duty - US'] = import_us
    air['Import Duty - UK'] = import_uk
    air['Import Duty - EU'] = import_eu
    #add fba and delivery
    sea['FBA Prep - US'] = unit_prep_us
    sea['FBA Prep - EU'] = unit_prep_eu
    sea['FBA Prep - UK'] = unit_prep_uk
    air['FBA Prep - US'] = unit_prep_us
    air['FBA Prep - EU'] = unit_prep_eu
    air['FBA Prep - UK'] = unit_prep_uk
    sea['FBA Delivery - US'] = unit_distribution_us
    sea['FBA Delivery - EU'] = unit_distribution_eu
    sea['FBA Delivery - UK'] = unit_distribution_uk
    air['FBA Delivery - US'] = unit_distribution_us
    air['FBA Delivery - EU'] = unit_distribution_eu
    air['FBA Delivery - UK'] = unit_distribution_uk 
    sea['Product Cost'] = product_cost_usd
    air['Product Cost'] = product_cost_usd
    #I would classify all this code under "it works and is reasonably performant so don't touch"
    sea['Cost of Capital - US'] = (sea['Product Cost'] +sea['Import Duty - US'] + sea['FBA Prep - US'] + sea['FBA Prep - US'] + sea['FBA Delivery - US'] + sea['Freight per Unit - US'])*cost_of_capital
    sea['Cost of Capital - EU'] = (sea['Product Cost'] +sea['Import Duty - EU'] + sea['FBA Prep - EU'] + sea['FBA Prep - EU'] + sea['FBA Delivery - EU'] + sea['Freight per Unit - EU'])*cost_of_capital
    sea['Cost of Capital - UK'] = (sea['Product Cost'] +sea['Import Duty - UK'] + sea['FBA Prep - UK'] + sea['FBA Prep - UK'] + sea['FBA Delivery - UK'] + sea['Freight per Unit - UK'])*cost_of_capital
    air['Cost of Capital - US'] = (air['Product Cost'] +air['Import Duty - US'] + air['FBA Prep - US'] + air['FBA Prep - US'] + air['FBA Delivery - US'] + air['Freight per Unit - US'])*cost_of_capital
    air['Cost of Capital - EU'] = (air['Product Cost'] +air['Import Duty - EU'] + air['FBA Prep - EU'] + air['FBA Prep - EU'] + air['FBA Delivery - EU'] + air['Freight per Unit - EU'])*cost_of_capital
    air['Cost of Capital - UK'] = (air['Product Cost'] +air['Import Duty - UK'] + air['FBA Prep - UK'] + air['FBA Prep - UK'] + air['FBA Delivery - UK'] + air['Freight per Unit - UK'])*cost_of_capital
    sea_us = sea[['Freight per Unit - US','Import Duty - US','FBA Prep - US','FBA Delivery - US','Product Cost','Cost of Capital - US']].copy()
    sea_uk = sea[['Freight per Unit - UK','Import Duty - UK','FBA Prep - UK','FBA Delivery - UK','Product Cost','Cost of Capital - UK']].copy()
    sea_eu = sea[['Freight per Unit - EU','Import Duty - EU','FBA Prep - EU','FBA Delivery - EU','Product Cost','Cost of Capital - EU']].copy()
    us_cols_to_sum = list(sea_us.columns)
    eu_cols_to_sum = list(sea_eu.columns)
    uk_cols_to_sum = list(sea_uk.columns)
    sea_us['Landed Cost - US'] = sea_us[us_cols_to_sum].sum(axis=1)
    sea_uk['Landed Cost - UK'] = sea_uk[uk_cols_to_sum].sum(axis=1)
    sea_eu['Landed Cost - EU'] = sea_eu[eu_cols_to_sum].sum(axis=1)
    air_us_final = air[['Freight per Unit - US','Import Duty - US','FBA Prep - US','FBA Delivery - US','Product Cost','Cost of Capital - US']].copy()
    air_uk_final = air[['Freight per Unit - UK','Import Duty - UK','FBA Prep - UK','FBA Delivery - UK','Product Cost','Cost of Capital - UK']].copy()
    air_eu_final = air[['Freight per Unit - EU','Import Duty - EU','FBA Prep - EU','FBA Delivery - EU','Product Cost','Cost of Capital - EU']].copy()
    us_cols_to_sum = list(air_us_final.columns)
    eu_cols_to_sum = list(air_eu_final.columns)
    uk_cols_to_sum = list(air_uk_final.columns)
    air_us_final['Landed Cost'] = air_us_final[us_cols_to_sum].sum(axis=1)
    air_uk_final['Landed Cost'] = air_uk_final[uk_cols_to_sum].sum(axis=1)
    air_eu_final['Landed Cost'] = air_eu_final[eu_cols_to_sum].sum(axis=1)
    sea_us = sea_us.T.reset_index().rename(columns={'index':'Attributes'})
    sea_eu = sea_eu.T.reset_index().rename(columns={'index':'Attributes'})
    sea_uk = sea_uk.T.reset_index().rename(columns={'index':'Attributes'})
    for x in [sea_us,sea_uk,sea_eu]:
        x['Method'] = 'Sea'
    sea_us['Marketplace'] = 'US'
    sea_eu['Marketplace'] = 'EU'
    sea_uk['Marketplace'] = 'UK'
    air_us_final = air_us_final.T.reset_index().rename(columns={'index':'Attributes'})
    air_eu_final = air_eu_final.T.reset_index().rename(columns={'index':'Attributes'})
    air_uk_final = air_uk_final.T.reset_index().rename(columns={'index':'Attributes'})
    for x in [air_us_final,air_uk_final,air_eu_final]:
        x['Method'] = 'Air'
    air_us_final['Marketplace'] = 'US'
    air_eu_final['Marketplace'] = 'EU'
    air_uk_final['Marketplace'] = 'UK'
    final_df =pd.concat([sea_us,sea_eu,sea_uk,air_us_final,air_eu_final,air_uk_final])
    final_df['Attributes'] = final_df['Attributes'].str.split('-').str[0]
    final_df.rename(columns={2:'USD'}, inplace=True)
    return final_df

def demand_planning(df_in,sspa_budget):
    x = full_year_generation(df_in)
    column_names = ['Months','Year','Full Date','Revenue','Units','ASP','Conversion','SSPA Conversion','GVs',
        'Organic Search GVs','Organic Search Units','SSPA GVs','SSPA Units','Other GVs','Other GV Units',
        'Category Rank','SSPA CPC','SSPA Budget','Total GVs','Total Organic Search GVs','Total SSPA GVs',
        'Total Other GVs','ACOS','TACOS','GV Share']
    y = pd.DataFrame(columns = column_names)
    #make 2 year projection and make it datetime
    y['Months'] = pd.Series(months())
    y = pd.concat([y]*2)
    y['Year'][0:12] = 2022
    y['Year'][12:24]= 2023
    launch_month_year = pd.to_datetime(str(launch_year) + launch_month, format = '%Y%b')
    y['Full Date'] = (pd.to_datetime(y['Year'].astype(str)  + y['Months'], format='%Y%b'))
    y = y[y['Full Date']>=launch_month_year]
    y.reset_index(inplace=True,drop=True)
    y=y.fillna(0)
    #Info on Total Market Size Pulled from H10 and put into main DF
    y['Total GVs'] = y['Months'].map(x.set_index('Month')['GVs'])
    y['Total Organic Search GVs'] = y['Months'].map(x.set_index('Month')['Organic Search GVs'])
    y['Total SSPA GVs'] = y['Months'].map(x.set_index('Month')['SSPA GVs'])
    y['Total Other GVs'] = y['Months'].map(x.set_index('Month')['Other GVs'])
    y['Conversion'] = conversion_percent

    #Sponsored Product Costs
    y['SSPA Conversion'] = conversion_percent
    y['SSPA CPC'] = sspa_cpc
    y['SSPA Budget'] = sspa_budget
    y.at[0,'SSPA Budget'] = 0
    y.at[1,'SSPA Budget'] = sspa_budget*2
    y.at[2,'SSPA Budget'] = sspa_budget*1.5
    y.at[3,'SSPA Budget'] = sspa_budget*1.25

    y['SSPA GVs'] = y['SSPA Budget']/y['SSPA CPC']
    y['SSPA Units'] = y['SSPA GVs']*y['SSPA Conversion']

    #Pricing - needs its own module
    y['ASP'] = y['Months'].map(x.set_index('Month')['ASP'])
    y.at[0,'ASP'] = 0
    y.at[1,'ASP'] = y.at[4,'ASP']/1.2
    y.at[2,'ASP'] = y.at[4,'ASP']/1.2
    y.at[3,'ASP'] = y.at[4,'ASP']/1.2

    
    #awful formula which takes the previous months sales and then applies them to next month to work out a category rank, then uses the exponent 
    #function we know to be roughly accurate to then derrive new sales.
    for index,row in y.iterrows():
        if index != 0:
            last_month = y.loc[index-1,'Organic Search Units']+y.loc[index-1,'SSPA Units']+y.loc[index-1,'Other GV Units']
            count =1
            last_month_sales = df_in['Sales']
            for x in last_month_sales:
                if x > last_month:
                    count = count +1
            exponent = exponenter(count)
            y.loc[index,'Category Rank'] = count
            y.loc[index,'Organic Search GVs'] = (0.25*y.loc[index,'Total Organic Search GVs'])*(count**exponent)
            y.loc[index,'Organic Search Units'] = round(y.loc[index,'Organic Search GVs']*y.loc[index,'Conversion'],0)
            y.loc[index,'Other GVs'] = (0.1*y.loc[index,'Total Other GVs'])*(count**exponent)
            y.loc[index,'Other GV Units'] = round(y.loc[index,'Other GVs']*y.loc[index,'Conversion'],0) 
            y.loc[index,'Units'] = round(y.loc[index,'Organic Search Units'],0)+round(y.loc[index,'SSPA Units'],0)+round(y.loc[index,'Other GV Units'],0)
    #Clean calculations for the rest after we've worked out the organic sales
    y['Revenue'] = y['Units'] * y['ASP']
    y['GVs'] = y['Organic Search GVs']+y['SSPA GVs']+y['Other GVs']
    y['TACOS'] = y['SSPA Budget'] / y['Revenue']
    y['ACOS'] = y['SSPA Budget']/(y['SSPA Units'] * y['ASP'])
    y['GV Share'] = y['GVs']/y['Total GVs']
    return y

def local_currency_pnl_builder(demand_sheet,sspa_budget,marketplace):

    #THIS CREATES THE LOCAL CURRENCY PNL BASED OFF THE MARKETPLACE
    y = demand_planning(demand_sheet,sspa_budget)
    y = y.drop(['Conversion','SSPA Conversion','GVs','Organic Search GVs',
        'Organic Search Units','SSPA GVs','SSPA Units','Other GVs','Other GV Units',
            'Total GVs','Total Organic Search GVs','Total SSPA GVs','Total Other GVs','GV Share',
            'SSPA CPC','ACOS','TACOS'],axis=1)
    landed_costs = cost_calculation(shipment_model,shipment_container)
    #converts usd freight cost to local marketplace to make sure it's a fully local currency pnl
    mp = marketplace
    if mp == 'US':
        currency_conversion = us_exchange_rate/us_exchange_rate #deliberately should resolve to 1
    elif mp == 'EU':
        currency_conversion = us_exchange_rate/eur_exchange_rate
    elif mp == 'UK':
        currency_conversion = us_exchange_rate
    #pulls in the shipment info calculated earlier by country and converts the costs to local currency
    filtered_info = landed_costs[landed_costs['Marketplace'] == mp]
    freight_per_unit = filtered_info[filtered_info['Attributes'].str.contains('Freight per Unit')]
    import_duty = filtered_info[filtered_info['Attributes'].str.contains('Import Duty')]
    import_duty = import_duty['USD'].values[0]*(1/currency_conversion)
    fba_prep = filtered_info[filtered_info['Attributes'].str.contains('FBA Prep')]
    fba_prep = fba_prep['USD'].values[0]*(1/currency_conversion)
    fba_delivery = filtered_info[filtered_info['Attributes'].str.contains('FBA Delivery')]
    fba_delivery = fba_delivery['USD'].values[0]*(1/currency_conversion)
    product_cost = filtered_info[filtered_info['Attributes'].str.contains('Product Cost')]
    product_cost = product_cost['USD'].values[0]*(1/currency_conversion)
    cost_of_capital = filtered_info[filtered_info['Attributes'].str.contains('Cost of Capital')]
    cost_of_capital = cost_of_capital['USD'].values[0]*(1/currency_conversion)
    landed_cost = filtered_info[filtered_info['Attributes'].str.contains('Landed Cost')]
    landed_cost = landed_cost['USD'].values[0]*(1/currency_conversion)

    #weighted average of air freight vs sea freight impact on pcogs
    freight_per_unit = np.average([freight_per_unit['USD'][freight_per_unit['Method']=='Air'].values[0],
        freight_per_unit['USD'][freight_per_unit['Method']=='Sea'].values[0]],weights=[air_percent,sea_percent])

    # builds the remaining parts of the PNL on both a per unit and total cost amount
    y['Revenue ex VAT'] = y['Revenue']/1.2 #PLACEHOLDER AT 20% VAT - needs to be a lookup table later
    y['Freight Costs per Unit'] = freight_per_unit
    y['Freight Costs Total'] = freight_per_unit*y['Units']
    y['Import Duty per Unit'] = import_duty
    y['Import Duty'] = import_duty*y['Units']
    y['FBA Prep per Unit'] = fba_prep
    y['FBA Prep'] = fba_prep*y['Units']
    y['FBA Delivery per Unit'] = fba_delivery
    y['FBA Delivery'] = fba_delivery*y['Units']
    y['Product Cost per Unit'] = product_cost
    y['Product Cost'] = product_cost*y['Units']
    y['Cost of Capital per Unit'] = cost_of_capital
    y['Cost of Capital'] = cost_of_capital*y['Units']
    y['Landed Cost per Unit'] = freight_per_unit+import_duty+fba_prep+fba_delivery+product_cost+cost_of_capital
    y['Landed Cost'] = (freight_per_unit+import_duty+fba_prep+fba_delivery+product_cost+cost_of_capital)*y['Units']
    y['PC1'] = y['Revenue ex VAT']-y['Landed Cost']
    y['PC1%'] = y['PC1']/y['Revenue ex VAT']
    y['Shipment Cost per Unit'] = amazon_holding_and_shipment_costs(marketplace) #placeholder
    y['Shipment Cost'] = y['Shipment Cost per Unit']*y['Units']
    y['Holding Cost per Unit'] = 0.25 #PLACEHOLDER - CAN WORK THIS OUT OFF VOLUME OF PACKAGE LATER
    y['Holding Cost'] = y['Holding Cost per Unit']*y['Units']
    y['PC2'] = y['PC1']-(y['Shipment Cost']+y['Holding Cost'])
    y['PC2%'] = y['PC2']/y['Revenue ex VAT']
    y['PC3'] = y['PC2']-y['SSPA Budget']
    y['PC3%'] = y['PC3']/y['Revenue ex VAT']
    return y


def amazon_holding_and_shipment_costs(marketplace):
    #super placeholder function - DATA IS IN NO WAY ACCURATE
    y = 3
    return y

def gbp_pnl(df_in,marketplace):
    #convert the columns to gbp for ease of upload into db etc
    y = local_currency_pnl_builder(df_in,sspa_budget,marketplace)
    if marketplace == "UK":
        y = y
    elif marketplace == "US":
        y.iloc[:,np.r_[3,5,7:24,25:30,31]] = y.iloc[:,np.r_[3,5,7:24,25:30,31]].mul(1/us_exchange_rate,axis=0)
    elif marketplace == "EU":
        y.iloc[:,np.r_[3,5,7:24,25:30,31]] = y.iloc[:,np.r_[3,5,7:24,25:30,31]].mul(1/eur_exchange_rate,axis=0)    
    return y

def initial_order_value(df_in,mp):
    mp = marketplace
    demand = demand_planning(df_in,sspa_budget)
    lead_time = (int(production_lead_time_months)+int(shipment_lead_time_months))
    launch_units = demand['Units'].iloc[1:lead_time+2].sum()
    return launch_units

#get dimensions and turn into cm where needed 
def package_dimensions(asin):
    package_height = product_info[product_info['asin']== asin]['amz_package_height'].values[0]
    package_height_unit = product_info[product_info['asin']== asin]['amz_package_height_unit'].values[0]
    package_width = product_info[product_info['asin']== asin]['amz_package_width'].values[0]
    package_width_unit = product_info[product_info['asin']== asin]['amz_package_height_unit'].values[0]
    package_length = product_info[product_info['asin']== asin]['amz_package_length'].values[0]
    package_length_unit = product_info[product_info['asin']== asin]['amz_package_length_unit'].values[0]
    package_weight = product_info[product_info['asin']== asin]['amz_package_weight'].values[0]
    package_weight_units = product_info[product_info['asin']== asin]['amz_package_weight_unit'].values[0]
    #convert inches to CM and lbs to kg
    if package_height_unit=='inches':
        package_height = (float(package_height)*float(2.54))/100
    if package_width_unit=='inches':
        package_width = (float(package_width)*float(2.54))/100
    if package_length_unit=='inches':
        package_length = (float(package_length)*float(2.54))/100
    if package_weight_units == 'pounds':
        package_weight = float(package_weight)*float(0.453592)
    return [package_height,package_width,package_length,package_weight]

def duty_costs(asin):
    uk_duty = product_info[product_info['asin']== asin]['duty_percentage_uk'].values[0]
    us_duty = product_info[product_info['asin']== asin]['duty_percentage_us'].values[0]
    eu_duty = product_info[product_info['asin']== asin]['duty_percentage_eu'].values[0]
    return [uk_duty, us_duty,eu_duty]

def product_costs(asin):
    local_cost = product_info[product_info['asin']== asin]['unit_cost_local'].values[0]
    cost_currency = product_info[product_info['asin']== asin]['unit_cost_currency'].values[0]
    if cost_currency =='USD':
        x = local_cost/us_exchange_rate
    if cost_currency =='EUR':
        x = local_cost/eur_exchange_rate
    if cost_currency =='GBP':
        x = local_cost
    if cost_currency =='AUD':
        x = local_cost*(0.53)#can't work out how to link AUD, only impacts 1 SKU
    return x

def seasonality_curve(asin):
    x = mb_query.product_seasonality('h')
    q = asin
    try:
        input_asins = str(q).split(',')
    except:
        input_asins = asin
    x = x[x['asin'].isin(input_asins)]
    x.reset_index(inplace=True,drop=True)
    x['revenuepercent'] = x['sum']/x['sum'].sum()
    x['revenue percent smoothed'] = x['revenuepercent'].rolling(window=3,min_periods=1, center=True).sum()
    x['revenue percent smoothed'] = x['revenue percent smoothed']/x['revenue percent smoothed'].sum()
    return x['revenue percent smoothed'].to_list()


asin = "B01FCMV5RM"

product_info = mb_query.product_download()
exchange_rate_query = mb_query.currency_rates()
us_exchange_rate = exchange_rate_query[exchange_rate_query['to_currency_code'] == 'USD']['avg'].values[0]
eur_exchange_rate = exchange_rate_query[exchange_rate_query['to_currency_code'] == 'EUR']['avg'].values[0]
product_cost_gbp = product_costs(asin)
product_cost_usd = product_cost_gbp*us_exchange_rate
product_cost_eur = product_cost_gbp*eur_exchange_rate

#global variables - ideally want these to be user defined at some point (or to be honest defined with data so it's automatic)
s_curve = [0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083,0.083]
price_discount_curve =[0,0,0,0,0,0,0,0,0,0,0,0]



#Freight Inputs - From user
sea_percent = 1
air_percent = 1-sea_percent
production_lead_time_months = np.ceil((product_info[product_info['asin']== asin]['production_lead_time'].values[0])/30)
shipment_lead_time_months = 3
lead_time = (int(production_lead_time_months)+int(shipment_lead_time_months))
launch_month = datetime.strptime(str(int((datetime.now().month-1+shipment_lead_time_months+production_lead_time_months))),"%m").strftime("%b")

print(production_lead_time_months,shipment_lead_time_months,launch_month)
launch_year = 2022


#ads cpc (local currency)
sspa_cpc = 1

#Marketplace - From User
marketplace = 'US'

#can be pulled from query
dimensions_list = package_dimensions(asin)
product_height = dimensions_list[0]
product_width = dimensions_list[1]
product_depth = dimensions_list[2]
unit_weight = dimensions_list[3]


#ideally need a query
duties = duty_costs(asin)
import_duty_us = duties[1]
import_duty_uk = duties[0]
import_duty_eu = duties[2]

#shipping costs (need to link to a query used by Kieran at some point)
shipment_model = 'Bear'
shipment_container ='40'
twenty_foot_us = [4000,7500,11000]
twenty_foot_uk = [4000,10000,11000]
twenty_foot_eu = [3500,7000,10000]
forty_foot_us = [5000,11500,16000]
forty_foot_uk = [7500,14500,20500]
forty_foot_eu = [6500,12500,18500]
forty_foot_high_cube_us = [6000,12000,17000]
forty_foot_high_cube_uk = [8000,15000,21000]
forty_foot_high_cube_eu = [7000,13000,19000]
air_us = [8,10,12]
air_uk = [7,9,11]
air_eu = [7,8,10]

#assumptions - probably don't want to surface to the user
organic_search_percent = 0.5
sspa_percent = 0.2
other_gv_percent = 1-(organic_search_percent+sspa_percent)
conversion_percent = 0.07



def dumb_maximise(df_in):
    ppc=[]
    pc3 = []
    for i in range(0,50000,250):
        x = local_currency_pnl_builder(df_in,i,marketplace)
        pc3.append(x['PC3'].iloc[1:13].sum())
        ppc.append(i)
    suggestions = tuple(zip(ppc,pc3))
    return max(suggestions,key=lambda item:item[1])[0]





es = h10_xray(r"C:\Users\jorda\Downloads\Helium_10_Xray_2022-02-07 (2).csv")

sspa_budget = dumb_maximise(es)

top_asin_full_year(es)
x = cost_calculation(shipment_model,shipment_container)
y = demand_planning(es,sspa_budget)
p = initial_order_value(es,marketplace)
z = local_currency_pnl_builder(es,sspa_budget,marketplace)
a = gbp_pnl(es,marketplace)
q = full_year_generation(es)
print(s_curve)
