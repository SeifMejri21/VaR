# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:32:35 2021

@author: Administrator
"""

import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm    
import seaborn as sns
import os


sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



#############################################################################################################
#############################################################################################################

### COTATION

cotation_2017 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\cotation2017.xlsx')
cotation_2018 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\cotation2018.xlsx')
cotation_2019 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\cotation2019.xlsx')
cotation_2020 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\cotation2020.xlsx')
cotation_2021 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\cotation2021.xlsx')

#############################################################################################################
#############################################################################################################


banks = ['AMEN BANK' , 'ATB', 'ATTIJARI BANK','BH', 'BIAT','BNA','BT','BTE','STB','UBCI','UIB','WIFACK INT BANK']

####Function that extracts banks data 

def banks_extracter(data):
    ###AMEN BANK
    AMEN_BANK_data= data[data['LIB_VAL'] =='AMEN BANK']
    AMEN_BANK_data=AMEN_BANK_data[['SEANCE','CLOTURE']]
    AMEN_BANK_data['SEANCE'] = pd.to_datetime(AMEN_BANK_data['SEANCE'])
    AMEN_BANK_data['SEANCE']=AMEN_BANK_data['SEANCE'].dt.strftime("%d/%m/%y")
    AMEN_BANK_data['SEANCE'] = pd.to_datetime(AMEN_BANK_data['SEANCE'])
    
    ###ATB
    ATB_data= data[data['LIB_VAL'] =='ATB']
    ATB_data=ATB_data[['SEANCE','CLOTURE']]
    ATB_data['SEANCE'] = pd.to_datetime(ATB_data['SEANCE'])
    ATB_data['SEANCE']=ATB_data['SEANCE'].dt.strftime("%d/%m/%y")
    ATB_data['SEANCE'] = pd.to_datetime(ATB_data['SEANCE'])     
    
    ###ATTIJARI BANK 
    ATTIJARI_BANK_data=data[data['LIB_VAL'] == 'ATTIJARI BANK']
    ATTIJARI_BANK_data=ATTIJARI_BANK_data[['SEANCE','CLOTURE']]
    ATTIJARI_BANK_data['SEANCE'] = pd.to_datetime(ATTIJARI_BANK_data['SEANCE'])
    ATTIJARI_BANK_data['SEANCE']=ATTIJARI_BANK_data['SEANCE'].dt.strftime("%d/%m/%y")
    ATTIJARI_BANK_data['SEANCE'] = pd.to_datetime(ATTIJARI_BANK_data['SEANCE'])
    
    
    ###BIAT
    BIAT_data=data[data['LIB_VAL'] == 'BIAT']
    BIAT_data=BIAT_data[['SEANCE','CLOTURE']]
    BIAT_data['SEANCE'] = pd.to_datetime(BIAT_data['SEANCE'])
    BIAT_data['SEANCE']=BIAT_data['SEANCE'].dt.strftime("%d/%m/%y")
    BIAT_data['SEANCE'] = pd.to_datetime(BIAT_data['SEANCE'])
    
    
    ###BH
    BH_data= data[data['LIB_VAL'] =='BH']
    BH_data=BH_data[['SEANCE','CLOTURE']]
    BH_data['SEANCE'] = pd.to_datetime(BH_data['SEANCE'])
    BH_data['SEANCE']=BH_data['SEANCE'].dt.strftime("%d/%m/%y")
    BH_data['SEANCE'] = pd.to_datetime(BH_data['SEANCE'])
    
    ###BT
    BT_data= data[data['LIB_VAL'] =='BT']
    BT_data=BT_data[['SEANCE','CLOTURE']]
    BT_data['SEANCE'] = pd.to_datetime(BT_data['SEANCE'])
    BT_data['SEANCE']=BT_data['SEANCE'].dt.strftime("%d/%m/%y")
    BT_data['SEANCE'] = pd.to_datetime(BT_data['SEANCE'])
    
    ###BNA
    BNA_data=data[data['LIB_VAL'] == 'BNA']
    BNA_data=BNA_data[['SEANCE','CLOTURE']]
    BNA_data['SEANCE'] = pd.to_datetime(BNA_data['SEANCE'])
    BNA_data['SEANCE']=BNA_data['SEANCE'].dt.strftime("%d/%m/%y")
    BNA_data['SEANCE'] = pd.to_datetime(BNA_data['SEANCE'])
    
    ###STB
    STB_data=data[data['LIB_VAL'] == 'STB']
    STB_data=STB_data[['SEANCE','CLOTURE']]
    STB_data['SEANCE'] = pd.to_datetime(STB_data['SEANCE'])
    STB_data['SEANCE']=STB_data['SEANCE'].dt.strftime("%d/%m/%y")
    STB_data['SEANCE'] = pd.to_datetime(STB_data['SEANCE'])
    
    ###UIB
    UIB_data= data[data['LIB_VAL'] =='UIB']
    UIB_data=UIB_data[['SEANCE','CLOTURE']]
    UIB_data['SEANCE'] = pd.to_datetime(UIB_data['SEANCE'])
    UIB_data['SEANCE']=UIB_data['SEANCE'].dt.strftime("%d/%m/%y")
    UIB_data['SEANCE'] = pd.to_datetime(UIB_data['SEANCE'])
    
    ###WIFACK INT BANK
    WIFACK_INT_BANK_data= data[data['LIB_VAL'] =='WIFACK INT BANK']
    WIFACK_INT_BANK_data=WIFACK_INT_BANK_data[['SEANCE','CLOTURE']]
    WIFACK_INT_BANK_data['SEANCE'] = pd.to_datetime(WIFACK_INT_BANK_data['SEANCE'])
    WIFACK_INT_BANK_data['SEANCE']=WIFACK_INT_BANK_data['SEANCE'].dt.strftime("%d/%m/%y")
    WIFACK_INT_BANK_data['SEANCE'] = pd.to_datetime(WIFACK_INT_BANK_data['SEANCE'])
    
    ###BTE
    BTE_data= data[data['LIB_VAL'] =='BTE (ADP)']
    BTE_data=BTE_data[['SEANCE','CLOTURE']]
    BTE_data['SEANCE'] = pd.to_datetime(BTE_data['SEANCE'])
    BTE_data['SEANCE']=BTE_data['SEANCE'].dt.strftime("%d/%m/%y")
    BTE_data['SEANCE'] = pd.to_datetime(BTE_data['SEANCE'])
    
    ###UBCI
    UBCI_data= data[data['LIB_VAL'] =='UBCI']
    UBCI_data=UBCI_data[['SEANCE','CLOTURE']]
    UBCI_data['SEANCE'] = pd.to_datetime(UBCI_data['SEANCE'])
    UBCI_data['SEANCE']=UBCI_data['SEANCE'].dt.strftime("%d/%m/%y")
    UBCI_data['SEANCE'] = pd.to_datetime(UBCI_data['SEANCE'])
    
    ALL_12_series= [AMEN_BANK_data['SEANCE'], AMEN_BANK_data['CLOTURE'],ATB_data['CLOTURE'],ATTIJARI_BANK_data['CLOTURE'],
                BH_data['CLOTURE'],BIAT_data['CLOTURE'],BNA_data['CLOTURE'],BT_data['CLOTURE'],BTE_data['CLOTURE'],
                STB_data['CLOTURE'],UBCI_data['CLOTURE'],UIB_data['CLOTURE'],WIFACK_INT_BANK_data['CLOTURE']]


    banks = ['AMEN BANK' , 'ATB', 'ATTIJARI BANK','BH', 'BIAT','BNA','BT','BTE','STB','UBCI','UIB','WIFACK INT BANK']
    
    ALL_12 = pd.DataFrame({'Date':ALL_12_series[0], 'AMEN BANK':ALL_12_series[1] ,'ATB':ALL_12_series[2] , 
                           'ATTIJARI BANK':ALL_12_series[3] , 'BH':ALL_12_series[4], 'BIAT':ALL_12_series[5], 
                           'BNA':ALL_12_series[6], 'BT':ALL_12_series[7], 'BTE':ALL_12_series[8],'STB':ALL_12_series[9], 
                           'UBCI':ALL_12_series[10], 'UIB':ALL_12_series[11] , 'WIFACK INT BANK' :ALL_12_series[12]  })
    
    
    ALL_12_a= ALL_12.apply(lambda x: pd.Series(x.dropna().values))
    data2 = ALL_12_a[banks]
    data2.index=ALL_12_a['Date']
    
    return(data2)

BQ_cotation_2017 = banks_extracter(cotation_2017)
BQ_cotation_2018 = banks_extracter(cotation_2018)
BQ_cotation_2019 = banks_extracter(cotation_2019)
BQ_cotation_2020 = banks_extracter(cotation_2020)
BQ_cotation_2021 = banks_extracter(cotation_2021)

weights_2 = np.array([0.049191223,0.034862453,0.109757111,0.040838577,0.144700953,0.088473734,0.301806458,0.014424502,0.095649049,0.037327708,0.073948058,0.009020174])

weights_1 = np.array([0.07 , 0.033 , 0.156 , 0.058 , 0.276 , 0.063  , 0.167 , 0.003 , 0.055 , 0.053 , 0.066 ])
weights_3 =  np.array([1])

weights_df = pd.DataFrame({'Banque': banks , 'Pondération' : weights_2})
weights_df = weights_df.transpose()
weights_df.to_excel(r'C:\Users\Administrator\Desktop\weights_dataframe.xlsx')


def VaR(data , conf_level , weights) :
    
    returns = data.pct_change()   
    cov_matrix = returns.cov()
    avg_rets = returns.mean()
    port_mean = avg_rets.dot(weights)
    port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    conf_level1 = conf_level
    from scipy.stats import norm    
    var_1d1 = norm.ppf(1-conf_level1)*port_stdev-port_mean
    #print("Value at Risk =", round(var_1d1*100,2),"%")
    return(var_1d1)


def CoVaR(data , conf_level , weights) :
    
    returns = data.pct_change()   
    cov_matrix = returns.cov()
    avg_rets = returns.mean()
    port_mean = avg_rets.dot(weights)
    port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    conf_level1 = conf_level
    from scipy.stats import norm    
    C_var_1d1= (norm.pdf(norm.ppf(conf_level1))*port_stdev)/conf_level1 - port_mean
    #print("Conditional Value at Risk =",round(C_var_1d1*100,2),"%")
    return(C_var_1d1)




data_list = [BQ_cotation_2017,BQ_cotation_2018,BQ_cotation_2019,BQ_cotation_2020,BQ_cotation_2021] 

years=[2017,2018,2019,2020,2021]
VaR_list=[0,0,0,0,0]
Co_VaR_list=[0,0,0,0,0]
banks_var_covar = []

for i in  range(5) :
    print('Year:' , i+2017)
    VaR(data_list[i],0.05 ,weights_2)
    CoVaR(data_list[i],0.05 ,weights_2)
    VaR_list[i]=VaR(data_list[i],0.05 , weights_2)
    Co_VaR_list[i]=CoVaR(data_list[i],0.05 , weights_2)
    

var_covar=pd.DataFrame({'year': years , 'VaR': VaR_list ,'CoVaR' : Co_VaR_list })

var_covar.to_excel(r'C:\Users\Administrator\Desktop\my_weights_var_covar.xlsx')



plt.style.use('ggplot')
plt.figure(figsize=(20, 14))
plt.plot(var_covar['year'], var_covar['VaR'], label = "VaR" , linestyle='--' , marker='o' , ms= 15)
plt.plot(var_covar['year'], var_covar['CoVaR'], label = "CVaR" ,linestyle='--' , marker='o' ,ms= 15)
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("Evolution of VaR / CoVaR ")
plt.savefig('VaR1.png')
plt.show()

#############################################################################################################
#############################################################################################################
#############################################################################################################

##### STaT DEs

BQ_cotation_2017_2021 = pd.concat(data_list)


def returns(data):
    returns = data.pct_change()
    returns_100=returns*100
    return(returns_100)

rendement_BQ= returns(BQ_cotation_2017_2021)



description_BQ= rendement_BQ.describe()
skew_BQ = BQ_cotation_2017_2021.skew()
kurt_BQ = BQ_cotation_2017_2021.kurt()

description_BQ.loc['skewness']=skew_BQ
description_BQ.loc['kurtosis']=kurt_BQ

description_BQ.to_excel(r'C:\Users\Administrator\Desktop\rendements_banks_description.xlsx')

BQ_cotation_2017_2021.to_excel(r'C:\Users\Administrator\Desktop\12banks.xlsx')


banks_var = pd.DataFrame({'AMEN BANK' : [0,0,0,0,0] , 'ATB' : [0,0,0,0,0], 'ATTIJARI BANK': [0,0,0,0,0], 'BH': [0,0,0,0,0], 'BIAT': [0,0,0,0,0],'BNA': [0,0,0,0,0],
                                'BT': [0,0,0,0,0],'BTE' : [0,0,0,0,0],'STB': [0,0,0,0,0],'UBCI': [0,0,0,0,0],'UIB': [0,0,0,0,0],'WIFACK INT BANK': [0,0,0,0,0]})

banks_covar = pd.DataFrame({'AMEN BANK' : [0,0,0,0,0] , 'ATB' : [0,0,0,0,0], 'ATTIJARI BANK': [0,0,0,0,0], 'BH': [0,0,0,0,0], 'BIAT': [0,0,0,0,0],'BNA': [0,0,0,0,0],
                                'BT': [0,0,0,0,0],'BTE' : [0,0,0,0,0],'STB': [0,0,0,0,0],'UBCI': [0,0,0,0,0],'UIB': [0,0,0,0,0],'WIFACK INT BANK': [0,0,0,0,0]})




for m in range(5):
    var=[]
    covarvar=[]
    for n in range(12) :
       banks_var.iloc[m,n] = VaR(data_list[m][[banks[n]]],0.05 ,weights_3)
       banks_covar.iloc[m,n] = CoVaR(data_list[m][[banks[n]]],0.05 ,weights_3)
    
    
banks_var.index=[2017,2018,2019,2020,2021]     
banks_covar.index=[2017,2018,2019,2020,2021]     
        
banks_var.to_excel(r'C:\Users\Administrator\Desktop\banks_VaR.xlsx')
banks_covar.to_excel(r'C:\Users\Administrator\Desktop\banks_Covar.xlsx')




#####VaR CoVaR 300

def VaR_300(data , conf_level , weights) :
    a=list(range(len(data.index)-300))
    for i in range(len(data.index)-300) : 
        data_rec = data.iloc[i:i+300,:]
        returns = data_rec.pct_change()   
        cov_matrix = returns.cov()
        avg_rets = returns.mean()
        port_mean = avg_rets.dot(weights)
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        conf_level1 = conf_level
        var_1d1 = norm.ppf(1-conf_level1)*port_stdev-port_mean
        #print("Conditional Value at Risk =",round(C_var_1d1*100,2),"%")
        a[i]=var_1d1
        var_1d1= 0
    return(a)  

def CoVaR_300(data , conf_level , weights) :
    a=list(range(len(data.index)-300))
    for i in range(len(data.index)-300) : 
        data_rec = data.iloc[i:i+300,:]
        returns = data_rec.pct_change()   
        cov_matrix = returns.cov()
        avg_rets = returns.mean()
        port_mean = avg_rets.dot(weights)
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        conf_level1 = conf_level
        C_var_1d1= (norm.pdf(norm.ppf(conf_level1))*port_stdev)/conf_level1 - port_mean
        #print("Conditional Value at Risk =",round(C_var_1d1*100,2),"%")
        a[i]=C_var_1d1
        C_var_1d1= 0
    return(a)


dates_300 = BQ_cotation_2017_2021.tail(820).index.to_list()
varss = VaR_300(BQ_cotation_2017_2021 , 0.05 , weights_2)
covars = CoVaR_300(BQ_cotation_2017_2021 , 0.05 , weights_2)

varss = [i * 100 for i in varss]
covars = [i * 100 for i in covars]


plt.style.use('ggplot')
plt.figure(figsize=(20, 12))
plt.plot(dates_300, varss , label = "VaR 300")
plt.plot(dates_300, covars, label = "CoVaR 300")
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("Evolution of VaR 300 / CoVaR 300")
plt.savefig('VaR1.png')
plt.show()


banks_var= pd.DataFrame(columns= banks)
banks_covar= pd.DataFrame(columns= banks)
banks_var.shape()
banks_covar.size()

for k in banks : 
    dates_300 = BQ_cotation_2017_2021[[k]].tail(820).index.to_list()
    varss = VaR_300(BQ_cotation_2017_2021[[k]] , 0.05 , weights_3)
    covars = CoVaR_300(BQ_cotation_2017_2021[[k]] , 0.05 , weights_3)
    
    varss = [i * 100 for i in varss]
    covars = [i * 100 for i in covars]
    banks_var[k] = varss
    banks_covar [k] =covars
    
    plt.style.use('ggplot')
    plt.figure(figsize=(20, 12))
    plt.plot(dates_300, varss , label = k+"  VaR 300")
    plt.plot(dates_300, covars, label = k+"  CoVaR 300")
    plt.legend()
    plt.xlabel("Year")
    plt.ylabel( k+  "VaR / CoVaR")
    plt.title( k+"  Evolution of VaR 300 / CoVaR 300")
    plt.show()


banks_var['date'] = dates_300
banks_covar['date'] = dates_300


##################################################################################################
##################################################################################################

pal = sns.color_palette(palette='coolwarm', n_colors=12)

# in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
g = sns.FacetGrid(banks_var, row='date', hue='ATB', aspect=15, height=0.75, palette=pal)

# then we add the densities kdeplots for each month
g.map(sns.kdeplot, 'Mean_TemperatureC',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

# here we add a white line that represents the contour of each kdeplot
g.map(sns.kdeplot, 'Mean_TemperatureC', 
      bw_adjust=1, clip_on=False, 
      color="w", lw=2)

# here we add a horizontal line for each plot
g.map(plt.axhline, y=0,
      lw=2, clip_on=False)

# we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
# notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
for i, ax in enumerate(g.axes.flat):
    ax.text(-15, 0.02, month_dict[i+1],
            fontweight='bold', fontsize=15,
            color=ax.lines[-1].get_color())
    
# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
g.fig.subplots_adjust(hspace=-0.3)

# eventually we remove axes titles, yticks and spines
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
plt.xlabel('Temperature in degree Celsius', fontweight='bold', fontsize=15)
g.fig.suptitle('Daily average temperature in Seattle per month',
               ha='right',
               fontsize=20,
               fontweight=20)
plt.show()




##################################################################################################
##################################################################################################









#####VaR CoVaR 50


def VaR_50(data , conf_level , weights) :
    a=list(range(len(data.index)-50))
    for i in range(len(data.index)-50) : 
        data_rec = data.iloc[i:i+50,:]
        returns = data_rec.pct_change()   
        cov_matrix = returns.cov()
        avg_rets = returns.mean()
        port_mean = avg_rets.dot(weights)
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        conf_level1 = conf_level
        var_1d1 = norm.ppf(1-conf_level1)*port_stdev-port_mean
        #print("Conditional Value at Risk =",round(C_var_1d1*100,2),"%")
        a[i]=var_1d1
        var_1d1= 0
    return(a) 
    a=list(range(len(data.index)-50))
    for i in range(len(data.index)-50) : 
        data_rec = data.iloc[i:i+50,:]
        returns = data_rec.pct_change()   
        cov_matrix = returns.cov()
        avg_rets = returns.mean()
        port_mean = avg_rets.dot(weights)
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        conf_level1 = conf_level
        var_1d1 = norm.ppf(1-conf_level1)*port_stdev-port_mean
        #print("Conditional Value at Risk =",round(C_var_1d1*100,2),"%")
        a[i]=var_1d1
        var_1d1= 0
    return(a)  

def CoVaR_50(data , conf_level , weights) :
    a=list(range(len(data.index)-50))
    for i in range(len(data.index)-50) : 
        data_rec = data.iloc[i:i+50,:]
        returns = data_rec.pct_change()   
        cov_matrix = returns.cov()
        avg_rets = returns.mean()
        port_mean = avg_rets.dot(weights)
        port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        conf_level1 = conf_level
        C_var_1d1= (norm.pdf(norm.ppf(conf_level1))*port_stdev)/conf_level1 - port_mean
        #print("Conditional Value at Risk =",round(C_var_1d1*100,2),"%")
        a[i]=C_var_1d1
        C_var_1d1= 0
    return(a)



dates_50 = BQ_cotation_2017_2021.tail(1070).index.to_list()
varss = VaR_50(BQ_cotation_2017_2021 , 0.05 , weights_2)
covars = CoVaR_50(BQ_cotation_2017_2021 , 0.05 , weights_2)

varss = [i * 100 for i in varss]
covars = [i * 100 for i in covars]


plt.style.use('ggplot')
plt.figure(figsize=(25, 12))
plt.plot(dates_50, varss , label = "VaR 50")
plt.plot(dates_50, covars, label = "CoVaR 50")
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("Evolution of VaR 50 / CoVaR 50")
plt.savefig('VaR1.png')
plt.show()





#############################################################################################################
#############################################################################################################
#############################################################################################################



##################################################################################################
##################################################################################################



##################################################################################################
##################################################################################################
##################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
 
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt



indice_2017 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\Indice2017.xlsx')
indice_2018 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\Indice2018.xlsx')
indice_2019 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\Indice2019.xlsx')
indice_2020 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\Indice2020.xlsx')
indice_2021 = pd.read_excel(r'C:\Users\Administrator\Desktop\hamidou\Indice2021.xlsx')



#################### INDICE bancaire 2017
BQ_indice_2017 = indice_2017[indice_2017['LIB_INDICE']=='INDBQ']
BQ_indice_2017=BQ_indice_2017[['SEANCE','INDICE_JOUR']]
BQ_indice_2017['SEANCE'] = pd.to_datetime(BQ_indice_2017['SEANCE'])
BQ_indice_2017['SEANCE']=BQ_indice_2017['SEANCE'].dt.strftime("%d/%m/%y")
BQ_indice_2017['SEANCE'] = pd.to_datetime(BQ_indice_2017['SEANCE'])
BQ_indice_2017.index=BQ_indice_2017['SEANCE']
BQ_indice_2017=BQ_indice_2017[['INDICE_JOUR']]




#################### INDICE bancaire 2018
BQ_indice_2018 = indice_2018[indice_2018['LIB_INDICE']=='INDBQ']
BQ_indice_2018=BQ_indice_2018[['SEANCE','INDICE_JOUR']]
BQ_indice_2018['SEANCE'] = pd.to_datetime(BQ_indice_2018['SEANCE'])
BQ_indice_2018['SEANCE']=BQ_indice_2018['SEANCE'].dt.strftime("%d/%m/%y")
BQ_indice_2018['SEANCE'] = pd.to_datetime(BQ_indice_2018['SEANCE'])
BQ_indice_2018.index=BQ_indice_2018['SEANCE']
BQ_indice_2018=BQ_indice_2018[['INDICE_JOUR']]





#################### INDICE bancaire 2019
BQ_indice_2019 = indice_2019[indice_2019['LIB_INDICE']=='INDBQ']
BQ_indice_2019=BQ_indice_2019[['SEANCE','INDICE_JOUR']]
BQ_indice_2019['SEANCE'] = pd.to_datetime(BQ_indice_2019['SEANCE'])
BQ_indice_2019['SEANCE']=BQ_indice_2019['SEANCE'].dt.strftime("%d/%m/%y")
BQ_indice_2019['SEANCE'] = pd.to_datetime(BQ_indice_2019['SEANCE'])
BQ_indice_2019.index=BQ_indice_2019['SEANCE']
BQ_indice_2019=BQ_indice_2019[['INDICE_JOUR']]



#################### INDICE bancaire 2020
BQ_indice_2020 = indice_2020[indice_2020['LIB_INDICE']=='INDBQ']
BQ_indice_2020=BQ_indice_2020[['SEANCE','INDICE_JOUR']]
BQ_indice_2020['SEANCE'] = pd.to_datetime(BQ_indice_2020['SEANCE'])
BQ_indice_2020['SEANCE']=BQ_indice_2020['SEANCE'].dt.strftime("%d/%m/%y")
BQ_indice_2020['SEANCE'] = pd.to_datetime(BQ_indice_2020['SEANCE'])
BQ_indice_2020.index=BQ_indice_2020['SEANCE']
BQ_indice_2020=BQ_indice_2020[['INDICE_JOUR']]



#################### INDICE bancaire 2021
BQ_indice_2021 = indice_2021[indice_2021['LIB_INDICE']=='INDBQ']
#BQ_indice_2021 = indice_2021[['SEANCE','INDICE_JOUR']]
BQ_indice_2021=BQ_indice_2021[['SEANCE','INDICE_JOUR']]
BQ_indice_2021['SEANCE'] = pd.to_datetime(BQ_indice_2021['SEANCE'])
BQ_indice_2021['SEANCE']=BQ_indice_2021['SEANCE'].dt.strftime("%d/%m/%y")
BQ_indice_2021['SEANCE'] = pd.to_datetime(BQ_indice_2021['SEANCE'])
BQ_indice_2021.index=BQ_indice_2021['SEANCE']
BQ_indice_2021=BQ_indice_2021[['INDICE_JOUR']]
BQ_indice_2021 = BQ_indice_2021.sort_values(by="SEANCE")

BQ_df_list = [BQ_indice_2017,BQ_indice_2018,BQ_indice_2019,BQ_indice_2020,BQ_indice_2021]
BQ_indice_2017_2021 = pd.concat(BQ_df_list)



plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.plot( BQ_indice_2017_2021['INDICE_JOUR'], label = "INDBQ")
plt.legend()
plt.xlabel("Year")
plt.ylabel("INDBQ")
plt.title("Evolution of TUNBANK")
plt.show()



years=[2017,2018,2019,2020,2021]
TUNBANK_VaR_list=[0,0,0,0,0]
TUNBANK_Co_VaR_list=[0,0,0,0,0]

data_list_BANK =[BQ_indice_2017,BQ_indice_2018,BQ_indice_2019,BQ_indice_2020,BQ_indice_2021]

for i in  range(5) :
    print('Year:' , i+2017)
    VaR(data_list_BANK[i],0.05 ,weights_3)
    CoVaR(data_list_BANK[i],0.05 ,weights_3)
    TUNBANK_VaR_list[i]=VaR(data_list_BANK[i],0.05 , weights_3)
    TUNBANK_Co_VaR_list[i]=CoVaR(data_list_BANK[i],0.05 , weights_3)
    

TUNBANK_VaR_list = [i * 100 for i in TUNBANK_VaR_list]
TUNBANK_Co_VaR_list = [i * 100 for i in TUNBANK_Co_VaR_list]
TUNBANK_var_covar=pd.DataFrame({'year': years , 'VaR': TUNBANK_VaR_list ,
                                'CoVaR' : TUNBANK_Co_VaR_list })

TUNBANK_var_covar.to_excel(r'C:\Users\Administrator\Desktop\TUNBANK_var_covar.xlsx')


TUNBANK_var_50 = VaR_50(BQ_indice_2017_2021, 0.05 , weights_3)
TUNBANK_covar_50 =CoVaR_50(BQ_indice_2017_2021, 0.05 , weights_3)
TUNBANK_var_300 =VaR_300(BQ_indice_2017_2021, 0.05 , weights_3)
TUNBANK_covar_300 =CoVaR_300(BQ_indice_2017_2021, 0.05 , weights_3)

TUNBANK_var_50 =  [i * 100 for i in TUNBANK_var_50]
TUNBANK_covar_50 = [i * 100 for i in TUNBANK_covar_50]
TUNBANK_var_300 = [i * 100 for i in TUNBANK_var_300]
TUNBANK_covar_300 = [i * 100 for i in TUNBANK_covar_300]


dates_300 = BQ_indice_2017_2021.tail(820).index.to_list()
dates_50 = BQ_indice_2017_2021.tail(1070).index.to_list()


TUNBANK_var_covar_50=pd.DataFrame({'Var50' : TUNBANK_var_50 , 'CoVar50' : TUNBANK_covar_50})
TUNBANK_var_covar_300=pd.DataFrame({'Var300' : TUNBANK_var_300 , 'CoVar300' : TUNBANK_var_300})

TUNBANK_var_covar_50.index = dates_50
TUNBANK_var_covar_300.index = dates_300


TUNBANK_var_covar_50.to_excel(r'C:\Users\Administrator\Desktop\TUNBANK_var_covar_50.xlsx')
TUNBANK_var_covar_300.to_excel(r'C:\Users\Administrator\Desktop\TUNBANK_var_covar_300.xlsx')


plt.style.use('ggplot')
plt.figure(figsize=(25, 14))
plt.plot(dates_50, TUNBANK_var_50, label = "TUNBANK VaR 50" )
plt.plot(dates_50, TUNBANK_covar_50, label = "TUNBANK CoVaR 50"  )
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("TUNBANK Evolution of VaR 50 / CoVaR 50 ")
plt.savefig('VaR1.png')
plt.show()


plt.style.use('ggplot')
plt.figure(figsize=(25, 14))
plt.plot(dates_300, TUNBANK_var_300, label = "TUNBANK VaR 300" )
plt.plot(dates_300, TUNBANK_covar_300, label = "TUNBANK CoVaR 300"  )
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("TUNBANK Evolution of VaR 300 / CoVaR 300 ")
plt.savefig('VaR1.png')
plt.show()







plt.style.use('ggplot')
plt.figure(figsize=(20, 14))
plt.plot(TUNBANK_var_covar['year'], TUNBANK_var_covar['VaR'], label = "TUNBANK VaR" , linestyle='--' , marker='o' , ms= 15)
plt.plot(TUNBANK_var_covar['year'], TUNBANK_var_covar['CoVaR'], label = "TUNBANK CVaR" , linestyle='--' , marker='o' , ms= 15)
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("TUNBANK Evolution of VaR / CoVaR ")
plt.savefig('VaR1.png')
plt.show()


####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################
####################################################################################################################################################################################################


#################### TUNINDEX 2017
tunindex_2017 = indice_2017[indice_2017['LIB_INDICE']=='TUNINDEX']
tunindex_2017=tunindex_2017[['SEANCE','INDICE_JOUR']]
tunindex_2017['SEANCE'] = pd.to_datetime(tunindex_2017['SEANCE'])
tunindex_2017['SEANCE']=tunindex_2017['SEANCE'].dt.strftime("%d/%m/%y")
tunindex_2017['SEANCE'] = pd.to_datetime(tunindex_2017['SEANCE'])
tunindex_2017.index=tunindex_2017['SEANCE']
tunindex_2017=tunindex_2017[['INDICE_JOUR']]





#################### TUNINDEX 2018
tunindex_2018 = indice_2018[indice_2018['LIB_INDICE']=='TUNINDEX']
tunindex_2018=tunindex_2018[['SEANCE','INDICE_JOUR']]
tunindex_2018['SEANCE'] = pd.to_datetime(tunindex_2018['SEANCE'])
tunindex_2018['SEANCE']=tunindex_2018['SEANCE'].dt.strftime("%d/%m/%y")
tunindex_2018['SEANCE'] = pd.to_datetime(tunindex_2018['SEANCE'])
tunindex_2018.index=tunindex_2018['SEANCE']
tunindex_2018=tunindex_2018[['INDICE_JOUR']]





#################### TUNINDEX 2019
tunindex_2019 = indice_2019[indice_2019['LIB_INDICE']=='TUNINDEX']
tunindex_2019=tunindex_2019[['SEANCE','INDICE_JOUR']]
tunindex_2019['SEANCE'] = pd.to_datetime(tunindex_2019['SEANCE'])
tunindex_2019['SEANCE']=tunindex_2019['SEANCE'].dt.strftime("%d/%m/%y")
tunindex_2019['SEANCE'] = pd.to_datetime(tunindex_2019['SEANCE'])
tunindex_2019.index=tunindex_2019['SEANCE']
tunindex_2019=tunindex_2019[['INDICE_JOUR']]



#################### TUNINDEX 2020
tunindex_2020 = indice_2020[indice_2020['LIB_INDICE']=='TUNINDEX']
tunindex_2020=tunindex_2020[['SEANCE','INDICE_JOUR']]
tunindex_2020['SEANCE'] = pd.to_datetime(tunindex_2020['SEANCE'])
tunindex_2020['SEANCE']=tunindex_2020['SEANCE'].dt.strftime("%d/%m/%y")
tunindex_2020['SEANCE'] = pd.to_datetime(tunindex_2020['SEANCE'])
tunindex_2020.index=tunindex_2020['SEANCE']
tunindex_2020=tunindex_2020[['INDICE_JOUR']]



#################### TUNINDEX 2021
tunindex_2021 = indice_2021[indice_2021['LIB_INDICE']=='TUNINDEX']
#BQ_indice_2021 = indice_2021[['SEANCE','INDICE_JOUR']]
tunindex_2021=tunindex_2021[['SEANCE','INDICE_JOUR']]
tunindex_2021['SEANCE'] = pd.to_datetime(tunindex_2021['SEANCE'])
tunindex_2021['SEANCE']=tunindex_2021['SEANCE'].dt.strftime("%d/%m/%y")
tunindex_2021['SEANCE'] = pd.to_datetime(tunindex_2021['SEANCE'])
tunindex_2021.index=tunindex_2021['SEANCE']
tunindex_2021=tunindex_2021[['INDICE_JOUR']]
tunindex_2021 = tunindex_2021.sort_values(by="SEANCE")


########################### TUNINDEX 17-21

tunindex_df_list = [tunindex_2017,tunindex_2018,tunindex_2019,tunindex_2020,tunindex_2021]
tunindex_2017_2021 = pd.concat(tunindex_df_list)



plt.style.use('ggplot')
plt.figure(figsize=(15, 5))
plt.plot( BQ_indice_2017_2021['INDICE_JOUR'], label = "TUNBANK")
plt.plot( tunindex_2017_2021['INDICE_JOUR'], label = "TUNINDEX")
plt.legend()
plt.xlabel("Year")
plt.ylabel("INDBQ / TUNINDEX")
plt.title("Evolution of TUNBANK & TUNINDEX")
plt.show()





years=[2017,2018,2019,2020,2021]
TUNindex_VaR_list=[0,0,0,0,0]
TUNindex_Co_VaR_list=[0,0,0,0,0]

data_list_index =[tunindex_2017,tunindex_2018,tunindex_2019,tunindex_2020,tunindex_2021]

for i in  range(5) :
    print('Year:' , i+2017)
    VaR(data_list_index[i],0.05 ,weights_3)
    CoVaR(data_list_index[i],0.05 ,weights_3)
    TUNindex_VaR_list[i]=VaR(data_list_index[i],0.05 , weights_3)
    TUNindex_Co_VaR_list[i]=CoVaR(data_list_index[i],0.05 , weights_3)
    

TUNindex_VaR_list = [i * 100 for i in TUNindex_VaR_list]
TUNindex_Co_VaR_list = [i * 100 for i in TUNindex_Co_VaR_list]
TUNindex_var_covar=pd.DataFrame({'year': years , 'VaR': TUNindex_VaR_list ,
                                'CoVaR' : TUNindex_Co_VaR_list })


TUNindex_var_covar.to_excel(r'C:\Users\Administrator\Desktop\TUNindex_var_covar.xlsx')


plt.style.use('ggplot')
plt.figure(figsize=(20, 14))
plt.plot(TUNindex_var_covar['year'], TUNindex_var_covar['VaR'], label = "TUNBINDEX VaR" , linestyle='--' , marker='o' , ms= 15)
plt.plot(TUNindex_var_covar['year'], TUNindex_var_covar['CoVaR'], label = "TUNBINDEX CVaR" , linestyle='--' , marker='o' , ms= 15)
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("TUNBINDEX Evolution of VaR / CoVaR ")
plt.savefig('VaR1.png')
plt.show()




plt.style.use('ggplot')
plt.figure(figsize=(20, 14))
plt.plot(TUNBANK_var_covar['year'], TUNBANK_var_covar['VaR'], label = "TUNBANK VaR", linestyle='--' , marker='o' , ms= 15)
plt.plot(TUNBANK_var_covar['year'], TUNBANK_var_covar['CoVaR'], label = "TUNBANK CVaR", linestyle='--' , marker='o' , ms= 15)
plt.plot(TUNindex_var_covar['year'], TUNindex_var_covar['VaR'], label = "TUNBINDEX VaR", linestyle='--' , marker='o' , ms= 15)
plt.plot(TUNindex_var_covar['year'], TUNindex_var_covar['CoVaR'], label = "TUNBINDEX CVaR", linestyle='--' , marker='o' , ms= 15)
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("TUNBANK & TUNBINDEX Evolution of VaR / CoVaR ")
plt.savefig('VaR1.png')
plt.show()






#########################################################################################################
#########################################################################################################
#########################################################################################################

dates_300 = tunindex_2017_2021.tail(820).index.to_list()
var_300_tunindex = VaR_300(tunindex_2017_2021 , 0.05 , weights_3)
co_var_300_tunindex = CoVaR_300(tunindex_2017_2021 , 0.05 , weights_3)
var_300_tunbank = VaR_300(BQ_indice_2017_2021 , 0.05 , weights_3)
co_var_300_tunbank = CoVaR_300(BQ_indice_2017_2021 , 0.05 , weights_3)



var_300_tunindex = [i * 100 for i in var_300_tunindex]
co_var_300_tunindex = [i * 100 for i in co_var_300_tunindex]
var_300_tunbank = [i * 100 for i in var_300_tunbank]
co_var_300_tunbank = [i * 100 for i in co_var_300_tunbank]



plt.style.use('ggplot')
plt.figure(figsize=(20, 12))
plt.plot(dates_300, var_300_tunindex , label = "TUNINDEX VaR 300")
plt.plot(dates_300, co_var_300_tunindex, label = "TUNINDEX CoVaR 300")
plt.plot(dates_300, var_300_tunbank , label = "TUNBANK VaR 300")
plt.plot(dates_300, co_var_300_tunbank, label = "TUNBANK CoVaR 300")
plt.legend()
plt.xlabel("Year")
plt.ylabel("VaR / CoVaR")
plt.title("TUNINDEX & TUNBANK Evolution of VaR 300 / CoVaR 300")
plt.savefig('VaR1.png')
plt.show()



#########################################################################################################
#########################################################################################################
#########################################################################################################

indexes_list = [BQ_indice_2017_2021 , tunindex_2017_2021]
indexes17_21 = pd.DataFrame({ 'TUNBANK': BQ_indice_2017_2021['INDICE_JOUR'] ,
                        'TUNINDEX': tunindex_2017_2021['INDICE_JOUR'] ,})

indexes17_21.index = BQ_indice_2017_2021.index


rendement_index= returns(indexes17_21)


description_index= rendement_index.describe()
skew_index = rendement_index.skew()
kurt_index = rendement_index.kurt()

description_index.loc['skewness']=skew_index
description_index.loc['kurtosis']=kurt_index

description_index.to_excel(r'C:\Users\Administrator\Desktop\rendements_inedex_description.xlsx')
indexes17_21.to_excel(r'C:\Users\Administrator\Desktop\inedex_17_21.xlsx')



plt.style.use('ggplot')
plt.figure(figsize=(30, 12))
plt.plot(rendement_index.index, rendement_index['TUNBANK'] , label = "TUNBANK rendements")
plt.plot(rendement_index.index, rendement_index['TUNINDEX'], label = "TUNINDEX rendements")
plt.legend()
plt.xlabel("Year")
plt.ylabel("rendements")
plt.title("TUNINDEX & TUNBANK rendements")
plt.savefig('VaR1.png')
plt.show()







