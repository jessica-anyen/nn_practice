"""
#som 動物圖:註解篇
-環境適用:nntest3
-參考資料:https://github.com/njali2001/popsom
-pandas 0.20.3
-py版本 3.5~3.6
-搭配popsom.py使用
#modify by jessica cheng & chu-chu lin 
"""
import popsom as som  
import pandas as pd
from   sklearn import datasets

#訓練資料
animal = ['dove','hen','duck','goose','owl','hawk','eagle','fox','dog','wolf','cat','tiger','lion','horse','zebra','cow']
attribute = [[1,0,0,1,0,0,1,0,1,0,0,1,0],
             [1,0,0,1,0,0,1,0,1,0,0,0,0],
             [1,0,0,1,0,0,1,0,1,0,0,0,1],
             [1,0,0,1,0,0,1,0,1,0,0,1,1],
             [1,0,0,1,0,0,1,0,1,1,0,1,0],
             [0,0,0,1,0,0,1,0,1,1,0,1,0],
             [0,1,0,1,0,0,1,0,1,1,0,1,0],
             [0,1,0,0,1,1,0,0,0,1,0,0,0],
             [0,1,0,0,1,1,0,0,0,0,1,0,0],
             [0,1,0,0,1,1,0,0,0,1,1,0,0],
             [1,0,0,0,1,1,0,0,0,1,0,0,0],
             [0,0,1,0,1,1,0,0,0,1,1,0,0],
			 [0,0,1,0,1,1,0,0,0,1,1,0,0],
			 [0,0,1,0,1,1,0,1,0,0,1,0,0],
			 [0,0,1,0,1,1,0,1,0,0,1,0,0],
             [0,0,1,0,1,1,0,0,0,0,0,0,0]]

attr = pd.DataFrame(attribute)
#欄位名稱
attr.columns = ['small','medium','big','2 legs','4 legs','hair','hooves','mane','feathers','hunt','run','fly','swim']
#初始化模型 #xdim, ydim 皆要大於3 #見popsom.py 53行
m = som.map(xdim=10,ydim=5)
#訓練資料
m.fit(attr,animal)
#呈現分群
m.starburst()