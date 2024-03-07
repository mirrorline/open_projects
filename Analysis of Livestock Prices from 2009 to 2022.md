# Analysis of Livestock Prices from 2009 to 2022
#### Importing Libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

#### Reading the Excel data
  ```python  
  file_path=r'C:\Users\Administrator\Desktop\Data Cleaning\Makert Prices 2022.xlsx'
    df=pd.read_excel(file_path)
    df
```
Rounding Off
```python    
df[['Bull','Cow','Heifer','Steer']]=df[['Bull','Cow','Heifer','Steer']].round(2)
```
Streamlining the Data
```python
df['Seasons'].unique()
#Output
        array(['Drought', 'Wet', 'Dry', 'dry', 'drought', 'wet'],dtype=object)
```

```python
df['Seasons']=df['Seasons'].replace('drought','Drought')
df['Seasons']=df['Seasons'].replace('dry','Dry')
df['Seasons']=df['Seasons'].replace('wet','Wet')
df
```
 
### Data Exploration
    df.info()
    df.head()
Output

    RangeIndex: 154 entries, 0 to 153
    Data columns (total 6 columns):
     #   Column   Non-Null Count  Dtype         
    ---  ------   --------------  -----         
     0   Dates    154 non-null    datetime64[ns]
     1   Seasons  154 non-null    object        
     2   Bull     154 non-null    float64       
     3   Cow      154 non-null    float64       
     4   Heifer   154 non-null    float64       
     5   Steer    154 non-null    float64       
    dtypes: datetime64[ns](1), float64(4), object(1)
    memory usage: 7.3+ KB


##### Seasons Overview
```python
df['Seasons'].value_counts()
```
    Seasons
    Dry        74
    Wet        55
    Drought    25
    Name: count, dtype: int64


**Analysing Prices by Seasons**
```python
#We declare variables to store data for the three seasons exclusively
drought_prices=df[df['Seasons']=='Drought']
dry_prices=df[df['Seasons']=='Dry']
wet_prices=df[df['Seasons']=='Wet']
```
Basic mean calculation
```python
mean_drought=drought_prices[['Bull','Cow','Heifer','Steer']].describe().round(2).loc['mean']
mean_wet=wet_prices[['Bull','Cow','Heifer','Steer']].describe().round(2).loc['mean']
mean_dry=dry_prices[['Bull','Cow','Heifer','Steer']].describe().round(2).loc['mean']
```
```python
    #Output
        Mean Wet prices 
     Bull      53882.40
    Cow       29382.87
    Heifer    25024.38
    Steer     42112.53
    Name: mean, dtype: float64 
    
     
         Mean Drought Prices 
    Bull      49801.24
    Cow       22672.80
    Heifer    17916.60
    Steer     38318.04
    Name: mean, dtype: float64 
    
         Mean Dry Prices
    Bull      54041.28
    Cow       28986.20
    Heifer    24942.11
    Steer     42223.96
    Name: mean, dtype: float64
```
**
```python
df[['Bull','Cow','Heifer','Steer']].describe().round(2).loc['std'].plot(kind='bar')

```

### Fitting a line Plot
```python
plt.figure(figsize=(16, 6))
plt.plot(df['Dates'], df['Bull'], marker='3', linestyle='-', label='Bull')
plt.plot(df['Dates'], df['Cow'], marker='3', linestyle='--', label='Cow')
plt.plot(df['Dates'], df['Heifer'], marker='3', linestyle='-.', label='Heifer')
plt.plot(df['Dates'], df['Steer'], marker='3', linestyle=':', label='Steer')
plt.xlabel('Date')
plt.ylabel('Prices by Livestock')
plt.title('Livestock Prices Over Time')
plt.grid(True)
plt.legend()
plt.show()
```
[Line Plot](https://github.com/mirrorline/open_projects/blob/main/linegraph.png)

### Correlation Analysis
```python
df_col=df[['Bull','Cow','Heifer','Steer']]
sns.pairplot(df_col)
plt.show()
```
[Correlation Plot](https://github.com/mirrorline/open_projects/blob/main/corr.png)


```python
plt.figure(figsize=(4,3))
sns.heatmap(df_col.corr(),annot=True, cmap='BrBG', fmt=".2f")
plt.title('Correlation Between Cattle Prices')
plt.show()
```
[HeatMap Plot](https://github.com/mirrorline/open_projects/commit/fe4d89692e748a6395d11f514c60ba61c6e9ed27#diff-452fa8ede054b0348d3b6e8a01fef988ff653d25e62ab78759395870f68f2cef)

From this plot, we see that the prices are generally correlated, with the Heifer and Steer having the least correlation

##  Modelling

```python
df['Year'] = df['Dates'].dt.year
df['Month'] = df['Dates'].dt.month
```


```python
X = df[['Year','Month']]
y = df[['Bull','Cow','Heifer','Steer']]
X_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
model=LinearRegression()
model.fit(X_train, y_train)
```
### Training for Predicitons

```python
train_predictions = model.predict(X_train)
test_predictions = model.predict(x_test)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

#Output
        Train RMSE: 10818.312165468164
        Test RMSE: 10054.704408210726
```

### Plotting Train and Test Predictions

```python
plt.figure(figsize=(10,6))
plt.scatter(y_test, test_predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('(Actual vs. Predicted) Livestock Prices')
plt.grid(True)
plt.show()
```
[Scatter Plot ](https://github.com/mirrorline/open_projects/blob/main/scatter%20plot.png)

