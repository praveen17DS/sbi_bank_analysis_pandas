import pandas as pd
import functools
import numpy as np



df=pd.read_csv("sbi.csv")
df.head()          
df = df[:-4]
df3 = df['Balance'].mask(df.isin(['', 0])).ffill()
df6 = df.assign(Balance=df3)
df7 = pd.to_numeric(df6['Balance'].str.replace(',',''), errors='coerce')
df8 = df.assign(Balance=df7)
df8


df8['grp'] = (df8.Balance != df8.Balance.shift()).cumsum()
out = df8.astype(str).groupby(['grp', 'Balance'])['Txn_Date','valueDate','Description','RefNoChequeNo','Debit','Credit'].agg(lambda x: ','.join(x.unique()))
out2 = out.reset_index()


out2.grp = pd.to_numeric(out2.grp, errors='coerce')
out2 = out2.sort_values(by=['grp'], ascending=[True])
out2 = out2.sort_values(by=['grp'], ascending=[True])
out3 = pd.to_numeric(out2['Balance'].str.replace('nan','NaN'), errors='coerce')
data4 = out2.assign(Balance=out3)
data = data4.dropna(subset=['Balance'])
data1 = data.drop(['grp'], axis=1)
data3 =  data1.reset_index(drop=True)


a1 = pd.to_numeric(data3['Txn_Date'].str.replace('nan',''), errors='coerce')


data3['Txn_Date'] = data3.Txn_Date.str.replace('nan,?' , '')
data3['Txn_Date'] = data3.Txn_Date.str.replace(',,?' , '-')
data3['valueDate'] = data3.valueDate.str.replace('nan,?' , '')
data3['valueDate'] = data3.valueDate.str.replace(',,?' , '-')
data3['Description'] = data3.Description.str.replace('nan,?' , '')
data3['Description'] = data3.Description.str.replace(',,?' , '-')
data3['RefNoChequeNo'] = data3.RefNoChequeNo.str.replace('nan,?' , '')
data3['RefNoChequeNo'] = data3.RefNoChequeNo.str.replace(',,?' , '-')
data3['Debit'] = data3.Debit.str.replace('nan,?' , '')
data3['Debit'] = data3.Debit.str.replace(',,?' , '-')
data3['Debit'] = data3.Debit.str.replace('-,?' , '')
data3['Credit'] = data3.Credit.str.replace('nan,?' , '')
data3['Credit'] = data3.Credit.str.replace(',,?' , '-')
data3['Credit'] = data3.Credit.str.replace('-,?' , '')


P = pd.to_datetime(data3['Txn_Date'], errors='coerce').dt.to_period('M')
data3['month_year'] = pd.to_datetime(data3['Txn_Date']).dt.to_period('M')




data4 = data3.replace(r'^\s*$', np.nan, regex=True)



data551 = data4[['month_year','Balance']]
data661 = data551.dropna()
ds1 = data661['Balance'].astype(str).astype(float)
Balance = data661.assign(Balance=ds1)
df32 = Balance.groupby('month_year').describe()


df32.reset_index()
df33 = Balance.groupby('month_year').sum()
df30=pd.merge(df32,df33,on="month_year",left_index=False)
df30=pd.merge(df32,df33,on="month_year",left_index=False)
df30.rename(columns = {('Balance','count'):'Balnace_Count'}, inplace = True) 
df30.rename(columns = {('Balance','mean'):'Balnace_Average'}, inplace = True) 
df30.rename(columns = {'Balance':'Balnace_Sum'}, inplace = True) 
df35=df30.drop(df30.columns[[2,3,4,5,6,7]], axis = 1)
data5 = data4[['month_year','Credit']]
data6 = data5.dropna()
ds = data6['Credit'].astype(str).astype(float)
Credit = data6.assign(Credit=ds)
g1 = pd.DataFrame(Credit.groupby('month_year').describe())
g2 = Credit.groupby('month_year').sum()


df31=pd.merge(g1,g2,on="month_year",left_index=False)
df31.rename(columns = {('Credit','count'):'Credit_Count'}, inplace = True) 
df31.rename(columns = {('Credit','mean'):'Credit_Average'}, inplace = True) 
df31.rename(columns = {'Credit':'Credit_Sum'}, inplace = True) 
df36=df31.drop(df31.columns[[2,3,4,5,6,7]], axis = 1)



data55 = data4[['month_year','Debit']]
data66 = data55.dropna()
ds3 = data66['Debit'].astype(str).astype(float)
Debit = data66.assign(Debit=ds3)
h1 = Debit.groupby('month_year').describe()

h2 = Debit.groupby('month_year').sum()

df32=pd.merge(h1,h2,on="month_year",left_index=False)
df32.rename(columns = {('Debit','count'):'Debit_Count'}, inplace = True) 
df32.rename(columns = {('Debit','mean'):'Debit_Average'}, inplace = True) 
df32.rename(columns = {'Debit':'Debit_Sum'}, inplace = True) 
# df32 = df32.reset_index(drop=True)
df37=df32.drop(df32.columns[[2,3,4,5,6,7]], axis = 1)



dfs = [df35, df36, df37]
df65_final = functools.reduce(lambda left,right: pd.merge(left,right,on='month_year'), dfs)

data443 = data3
data443.head(20)
data443['year_month_day'] = pd.to_datetime(data443['Txn_Date']).dt.to_period('D')
year = data443.year_month_day.dt.year
data443['year'] = year

month = data443.year_month_day.dt.month
data443['month'] = month

day = data443.year_month_day.dt.day
data443['day'] = day
data44 = data443




initial = 1
min_year = min(data44['year'])
d = data44.loc[(data44['year'] == min_year),["month"]]
min_month = min(d['month'])

max_year = max(data44['year'])
d = data44.loc[(data44['year'] == max_year),["month"]]
max_month = max(d['month'])


data44["status"]=0
start_year = min_year
end_year = max_year
start_month = min_month
end_month = max_month
flag = 0
while (start_year <=end_year):
    if flag == 0:
        start_month = min_month
    else:
        start_month = 1
    while(start_month <= 12):
        flag = 1
        data44.loc[(data44["month"] == start_month) & (data44['year'] == start_year),["status"]] = initial
        start_month = start_month + 1
        initial = initial + 1
    start_year = start_year + 1    



j1 = data44['status'].tail(1)
j = int(j1+1)
start = min(data44['status'])
end = max(data44['status'])
print(start)
p=1
q=10
r=20
a_final=pd.DataFrame(columns=['Balance','Txn_Date','valueDate','Description','RefNoChequeNo','Debit','Credit','month_year','year_month_day','year','month','day','status'])
b_final=pd.DataFrame(columns=['Balance','Txn_Date','valueDate','Description','RefNoChequeNo','Debit','Credit','month_year','year_month_day','year','month','day','status'])
c_final=pd.DataFrame(columns=['Balance','Txn_Date','valueDate','Description','RefNoChequeNo','Debit','Credit','month_year','year_month_day','year','month','day','status'])
for i in range(1,j):
    a=0
    b=0
    p=1
    q=10
    r=20
    if i == start:
        a = data44[((data44['status']==i) & (data44['day'] == p))].tail(1)
        if len(a)==0:
            a=data44[((data44['status']==i))].head(1)
    else:
        a = data44[((data44['status']==i) & (data44['day'] == p))].tail(1)
        if len(a)==0:
            a = data44[(data44['status']==i-1)].tail(1)
           
    a_final = pd.concat([a, a_final], ignore_index=True)
       
    fg=0       
    b = data44.loc[((data44['status']==i) & (data44['day'] == q))].tail(1)
    while (len(b)==0) & (fg==0):
        b = data44[((data44['status']==i) &(data44['day']==max((data44['day'])[(data44['day']==q)])))].tail(1)
        q = q-1
        if (q != 0) :
            continue
        elif (len(b)==0):
            b = data44[(data44['status']==i-1)].tail(1)
               
           
         
    b_final = pd.concat([b, b_final], ignore_index=True)
      
       
    fg1=0       
    c = data44.loc[((data44['status']==i) & (data44['day'] == r))].tail(1)
    while (len(c)==0) & (fg1==0):
        c = data44[((data44['status']==i) &(data44['day']==max((data44['day'])[(data44['day']==r)])))].tail(1)
        r = r-1
        if (r != 0) :
            continue
        elif (len(c)==0):
            c = data44data44[(data44['status']==i-1)].tail(1)
               
           
         
    c_final = pd.concat([c, c_final], ignore_index=True) 




final_final = pd.concat([a_final,b_final, c_final], ignore_index=True)
final_final=final_final.sort_values(by=['month_year','day'])



final_final = final_final.reset_index(drop=True)
final1 = final_final.reset_index(drop=False)
final2 = final1.drop(['index'], axis = 1) 
final2.reset_index(inplace=True)
groups = final2['Balance'].groupby(np.arange(len(final2.index))//3)
aa = final2.reset_index()
 
 
index1 =pd.DataFrame(aa['index'].iloc[np.arange(len(aa)/3).repeat(3)])
index1 = index1.rename(columns={"index": "actl_index"})
 
aa1 = aa.assign(index1=index1.values)
aa1.set_index('index1', inplace=True)
aa2 = pd.DataFrame(aa1.groupby(['index1'])['Balance'].apply(lambda x: ','.join(x.astype(str))).reset_index())

 
aa2[['1st_Bal','10th_Bal','20th_Bal']] = aa2.Balance.str.split(",",expand=True)
aa3 = aa2.iloc[:,2:]
df65_final[['1st_Bal','10th_Bal','20th_Bal']] = aa3
df65_final[['1st_Bal','10th_Bal','20th_Bal']] = aa3.to_numpy()

column_names=['1st_Bal','10th_Bal','20th_Bal','Balnace_Count','Balnace_Average','Balnace_Sum','Credit_Count','Credit_Average','Credit_Sum','Debit_Count','Debit_Average','Debit_Sum']
df65_final = df65_final.reindex(columns=column_names)
print(df65_final.reset_index())