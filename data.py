# 处理数据集的代码
# 选择的数据集："amazon_co-ecommerce-sample.csv"
import numpy as np
import pandas as pd

df = pd.read_csv("amazon_co-ecommerce_sample.csv")
df1 = df[['product_name', 'manufacturer', 'price', 'number_of_reviews', 'average_review_rating', 'amazon_category_and_sub_category']]
df1 = df1.dropna()

df1['number_of_reviews'] = df1['number_of_reviews'].replace(regex=True, to_replace=r',', value=r'')
df1.number_of_reviews = pd.to_numeric(df1['number_of_reviews'])

df1['price'] = df1['price'].replace(regex=True, to_replace=r'£', value=r'')
df1.drop(df1[df1['price'].str.contains(pat='-',regex=False)].index,inplace=True)  #regex=True则pat是一个正则表达式，regex=False表示pat是一个字符串
df1['price'] = df1['price'].replace(regex=True, to_replace=r',', value=r'')
df1['price'] = pd.to_numeric(df1['price'])

df1['percent'] = df1['number_of_reviews'] / df1['number_of_reviews'].sum()







