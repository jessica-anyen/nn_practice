"""
#som 動物圖:註解篇
-環境適用:nntest3
-參考資料:https://github.com/njali2001/popsom
-pandas 0.20.3
-py版本 3.5~3.6
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')  #不註解會跑不出圖
import matplotlib.pyplot as plt
import seaborn as sns					
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
import statsmodels.stats.api as sms     # t-test
import statistics as stat               # F-test
from scipy import stats                 # KS Test
from scipy.stats import f               # F-test
from itertools import combinations
import popsom as som  

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

##
class map:
	def __init__(self, xdim=10, ydim=5, alpha=.3, train=1000, norm=False):
		""" __init__ -- Initialize the Model 初始化模型

			parameters:
			- xdim,ydim - map的維度，x,y方向各多少神經元
			- alpha - 學習率, 需要是非零正整數
			- train - 訓練的次數
			- algorithm - 選擇演算法 (som and som_f)
			- norm - 正規化輸入資料
    	"""
		self.xdim = xdim
		self.ydim = ydim
		self.alpha = alpha
		self.train = train
		self.norm = norm

	def fit(self, data, labels):
		""" fit -- Train the Model with Python 訓練模型

			parameters:
			- data - 具有未標記實例的data
			- labels - 具有一個標記的資料行的向量
    	"""

		if self.norm:
    		#按列把data除以該行總和
			data = data.div(data.sum(axis=1), axis=0)
			
		self.data = data	
		self.labels = labels

		# 檢查神經元是否合理
		if (self.xdim < 3 or self.ydim < 3):
			sys.exit("build: map is too small.")

		self.vsom_p()

		visual = []

		for i in range(self.data.shape[0]):
			b = self.best_match(self.data.iloc[[i]])
			visual.extend([b])

		self.visual = visual
		
	def marginal(self, marginal):
		""" marginal -- 印出顯示神經元與資料的邊界機率分布

		 	parameters:
		 	- marginal:訓練資料的維度或索引即為marginal
		"""
		
		# check if the second argument is of type character
		if type(marginal) == str and marginal in list(self.data):

			f_ind = list(self.data).index(marginal)
			f_name = marginal
			train = np.matrix(self.data)[:, f_ind]
			neurons = self.neurons[:, f_ind]
			plt.ylabel('Density')
			plt.xlabel(f_name)
			#kdeplot:核密度估計圖，可以比較直觀的看出數據樣本本身的分布特徵
			sns.kdeplot(np.ravel(train),
				        label="training data",
						shade=True,
						color="b")
			sns.kdeplot(neurons, label="neurons", shade=True, color="r")
			plt.legend(fontsize=15)
			plt.show()

		elif (type(marginal) == int and marginal < len(list(self.data)) and marginal >= 0):

			f_ind = marginal
			f_name = list(self.data)[marginal]
			train = np.matrix(self.data)[:, f_ind]
			neurons = self.neurons[:, f_ind]
			plt.ylabel('Density')
			plt.xlabel(f_name)
			sns.kdeplot(np.ravel(train),
						label="training data",
						shade=True,
						color="b")
			sns.kdeplot(neurons, label="neurons", shade=True, color="r")
			plt.legend(fontsize=15)
			plt.show()

		else:
			sys.exit("marginal: second argument is not the name of a training \
						data frame dimension or index")

	def vsom_p(self):
		""" vsom_p -- 向量的，隨機som未優化的版本
		
    	"""
    	# some constants 一些參數
		dr = self.data.shape[0]
		dc = self.data.shape[1]
		nr = self.xdim*self.ydim
		nc = dc  # 資料維度與神經元數量是相同的

	    # 建立與初始化神經元
		cells = nr * nc  # 神經元數量 乘上 資料維度

	    # 具有神經元初始值的向量
		v = np.random.uniform(-1, 1, cells)

	    # 每行代表神經元，每列代表維度 
		neurons = np.transpose(np.reshape(v, (nc, nr)))  # rearrange the vector as matrix

		# neurons = np.reshape(v, (nr, nc)) # Another option to reshape

	    # 計算初始nsize(鄰居大小)和每一步驟(nsize_step)
		nsize = max(self.xdim, self.ydim) + 1
		nsize_step = np.ceil(self.train/nsize)
		step_counter = 0  # 計算每一個step的epoch數量

	    # 轉換一維列座標到二維圖 
		def coord2D(rowix):

			x = np.array(rowix) % self.xdim
			y = np.array(rowix) // self.xdim

			return np.concatenate((x, y))

	    # gamma函數的常數
		m = [i for i in range(nr)]  # 這個向量有所以有神經元的一維位址

	    # x-y coordinate of ith neuron: m2Ds[i,] = c(xi, yi)
		m2Ds = np.matrix.transpose(coord2D(m).reshape(2, nr))

	    # neighborhood function 鄰居函數
		def Gamma(c):

	        # lookup the 2D map coordinate for c
			c2D = m2Ds[c, ]
	        # a matrix with each row equal to c2D
			c2Ds = np.outer(np.linspace(1, 1, nr), c2D)
	        # distance vector of each neuron from c in terms of map coords!
			d = np.sqrt(np.dot((c2Ds - m2Ds)**2, [1, 1]))
	        # if m on the grid is in neigh then alpha else 0.0
			hood = np.where(d < nsize*1.5, self.alpha, 0.0)

			return hood
		
	    # training #訓練
	    # the epochs loop
		
		self.animation = []

		for epoch in range(self.train):

	        # hsize 隨nsize.steps減少 
			step_counter = step_counter + 1
			if step_counter == nsize_step:

				step_counter = 0
				nsize = nsize - 1

	        # 建立樣本訓練向量 
			ix = randint(0, dr-1)
			# ix = (epoch+1) % dr   # For Debugging
			xk = self.data.iloc[[ix]]

	        # 競爭
			xk_m = np.outer(np.linspace(1, 1, nr), xk)
			#計算神經元與cluster中心的距離diff
			diff = neurons - xk_m
			# 取平方
			squ = diff * diff
			s = np.dot(squ, np.linspace(1, 1, nc))
			o = np.argsort(s)
			c = o[0]

	        # 更新:神經元與中心的距離
			gamma_m = np.outer(Gamma(c), np.linspace(1, 1, nc))
			neurons = neurons - diff * gamma_m

			self.animation.append(neurons.tolist())
		
		self.neurons = neurons
		
	def convergence(self, conf_int=.95, k=50, verb=False, ks=False):
		""" convergence -- map的收斂
		
			Parameters:
			- conf_int - 信賴區間 (default 95%)
			- k - the number of samples used for the estimated topographic accuracy computation
			- verb - if true reports, 兩者各自收斂 
						otherwise, 對兩者線性結合
			- ks - if true: ks-test, 
			       if false:standard var and means test
			
			Return
			- 回傳這個值是不是收斂index 
		"""

		if ks:
			embed = self.embed_ks(conf_int, verb=False)
		else:
			embed = self.embed_vm(conf_int, verb=False)

		topo_ = self.topo(k, conf_int, verb=False, interval=False)

		if verb:
			return {"embed": embed, "topo": topo_} #各自
		else:
			return (0.5*embed + 0.5*topo_)	#將兩者結合	

	def starburst(self, explicit=False, smoothing=2, merge_clusters=True,  merge_range=.25):
		""" starburst -- 計算並呈現clusters結果
			
			parameters:
			- explicit - 控制連結單元的形狀
			- smoothing - 控制umat的滑順程度 (NULL,0,>0)
			- merge_clusters - 控制cluster是否merged together
			- merge_range - 定義多近才要跟中心merge，定義是data是靠近自己的群集還是別人的群集 
		"""

		umat = self.compute_umat(smoothing=smoothing)
		self.plot_heat(umat,
						explicit=explicit,
						comp=True,
						merge=merge_clusters,
						merge_range=merge_range)

	def compute_umat(self, smoothing=None):
		""" compute_umat -- 計算統一(unified)距離矩陣
		
			parameters:
			- smoothing - is either NULL, 0, or a positive floating point value controlling the
			              smoothing of the umat representation
			return:
			- a matrix with the same x-y dims as the original map containing the umat values
		"""

		d = euclidean_distances(self.neurons, self.neurons)
		umat = self.compute_heat(d, smoothing)

		return umat

	def compute_heat(self, d, smoothing=None):
		""" compute_heat -- 計算熱值map，表示距離矩陣
			
			parameters:
			- d - 使用'dist'函數計算出來的距離
			- smoothing - is either NULL, 0, or a positive floating point value controlling the
			        	  smoothing of the umat representation
			
			return:
			- a matrix with the same x-y dims as the original map containing the heat
		"""

		x = self.xdim
		y = self.ydim
		heat = np.matrix([[0.0] * y for _ in range(x)])

		if x == 1 or y == 1:
			sys.exit("compute_heat: heat map can not be computed for a map \
	                 with a dimension of 1")

		# 這個函數轉換成2-維 map
		# 使用一維神經元
		def xl(ix, iy):

			return ix + iy * x

		# 檢查map是不是2*2
		if x > 2 and y > 2:
			# iterate over the inner nodes and compute their umat values
			for ix in range(1, x-1):
				for iy in range(1, y-1):
					sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
						   d[xl(ix, iy), xl(ix, iy-1)] +
	                       d[xl(ix, iy), xl(ix+1, iy-1)] +
	                       d[xl(ix, iy), xl(ix+1, iy)] +
	                       d[xl(ix, iy), xl(ix+1, iy+1)] +
	                       d[xl(ix, iy), xl(ix, iy+1)] +
	                       d[xl(ix, iy), xl(ix-1, iy+1)] +
	                       d[xl(ix, iy), xl(ix-1, iy)])

					heat[ix, iy] = sum/8

			# 轉換 bottom x axis
			for ix in range(1, x-1):
				iy = 0
				sum = (d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix+1, iy+1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

			# 轉換 top x axis
			for ix in range(1, x-1):
				iy = y-1
				sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

			# 轉換 左 y-axis
			for iy in range(1, y-1):
				ix = 0
				sum = (d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy-1)] +
	                   d[xl(ix, iy), xl(ix+1, iy)] +
	                   d[xl(ix, iy), xl(ix+1, iy+1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)])

				heat[ix, iy] = sum/5

			# 轉換 右 y-axis
			for iy in range(1, y-1):
				ix = x-1
				sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy-1)] +
	                   d[xl(ix, iy), xl(ix, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy+1)] +
	                   d[xl(ix, iy), xl(ix-1, iy)])

				heat[ix, iy] = sum/5

		# 計算 umat values for corners
		if x >= 2 and y >= 2:
			# 左下角
			ix = 0
			iy = 0
			sum = (d[xl(ix, iy), xl(ix+1, iy)] +
	               d[xl(ix, iy), xl(ix+1, iy+1)] +
	               d[xl(ix, iy), xl(ix, iy+1)])

			heat[ix, iy] = sum/3

			# 右下角
			ix = x-1
			iy = 0
			sum = (d[xl(ix, iy), xl(ix, iy+1)] +
	               d[xl(ix, iy), xl(ix-1, iy+1)] +
	               d[xl(ix, iy), xl(ix-1, iy)])
			heat[ix, iy] = sum/3

			# 左上角
			ix = 0
			iy = y-1
			sum = (d[xl(ix, iy), xl(ix, iy-1)] +
	               d[xl(ix, iy), xl(ix+1, iy-1)] +
	               d[xl(ix, iy), xl(ix+1, iy)])
			heat[ix, iy] = sum/3

			# 右上角
			ix = x-1
			iy = y-1
			sum = (d[xl(ix, iy), xl(ix-1, iy-1)] +
	               d[xl(ix, iy), xl(ix, iy-1)] +
	               d[xl(ix, iy), xl(ix-1, iy)])
			heat[ix, iy] = sum/3

		# 讓heat map平滑
		pts = []

		for i in range(y):
			for j in range(x):
				pts.extend([[j, i]])

		if smoothing is not None:
			if smoothing == 0:
				heat = self.smooth_2d(heat,
									  nrow=x,
									  ncol=y,
									  surface=False)
			elif smoothing > 0:
				heat = self.smooth_2d(heat,
									  nrow=x,
									  ncol=y,
									  surface=False,
									  theta=smoothing)
			else:
				sys.exit("compute_heat: bad value for smoothing parameter")

		return heat

	def plot_heat(self, heat, explicit=False, comp=True, merge=False, merge_range=0.25):
		""" plot_heat -- 畫出熱度圖

			parameters:
			- heat - 2D heat map
			- explicit - 控制連結元件的形狀
			- comp - 控制是否連結元件
			- merge - 控制如何合併群集
			- merge_range - 定義一個範圍，定義元素比較接近自己的中心還是別人的中心
		"""

		umat = heat

		x = self.xdim
		y = self.ydim
		nobs = self.data.shape[0]
		count = np.matrix([[0]*y]*x)

		# 確保維度不是1
		if (x <= 1 or y <= 1):
			sys.exit("plot_heat: map dimensions too small")

		tmp = pd.cut(heat, bins=100, labels=False)
		
		tmp_1 = np.array(np.matrix.transpose(tmp))
		
		fig, ax = plt.subplots()
		#ax.pcolor(tmp_1, cmap=plt.cm.YlOrRd) #YlOrRd是紅黃配色
		ax.pcolor(tmp_1, cmap=plt.cm.spring) #這裡可以換底圖顏色
		"""
		#可用顏色
		['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
		('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),				 
		"""

		ax.set_xticks(np.arange(x)+0.5, minor=False)
		ax.set_yticks(np.arange(y)+0.5, minor=False)
		plt.xlabel("x")
		plt.ylabel("y")
		ax.set_xticklabels(np.arange(x), minor=False)
		ax.set_yticklabels(np.arange(y), minor=False)
		ax.xaxis.set_tick_params(labeltop='on')
		ax.yaxis.set_tick_params(labelright='on')

		# 將連結線放在map上
		if comp:
			if not merge:
				# 找map上每一個神經元的中心點
				centroids = self.compute_centroids(heat, explicit)
			else:
				# 找map上唯一的中心點
				centroids = self.compute_combined_clusters(umat, explicit, merge_range)

			# 將神經元連接到他的中心點
			for ix in range(x):
				for iy in range(y):
					cx = centroids['centroid_x'][ix, iy]
					cy = centroids['centroid_y'][ix, iy]
					plt.plot([ix+0.5, cx+0.5],
	                         [iy+0.5, cy+0.5],
	                         #color='grey',
	                         color='blue',   #這邊可以改線的顏色
							 linestyle='-',
	                         linewidth=1.0)

		# 將標籤放上map，如果有的話
		if not (self.labels is None) and (len(self.labels) != 0):

			# 計算每一個map cell有幾個label
			for i in range(nobs):

				nix = self.visual[i]
				c = self.coordinate(nix)
				ix = c[0]
				iy = c[1]

				count[ix-1, iy-1] = count[ix-1, iy-1]+1

			for i in range(nobs):

				c = self.coordinate(self.visual[i])
				ix = c[0]
				iy = c[1]

				# 每一個cell只印一個label
				#print(count[ix-1, iy-1])
				"""
				# we only print one label per cell
				if count[ix-1, iy-1] > 0:
				#if count[ix-1, iy-1] > 1:

					count[ix-1, iy-1] = 0
					ix = ix - .5
					iy = iy - .5
					l = self.labels[i]
					plt.text(ix+1, iy+1, l)
				
					# jc add for 顯示兩個label
				"""	
			  	# we only print one label per cell
				if count[ix-1, iy-1] > 1:

					count[ix-1, iy-1] -= 1
					ix = ix
					iy = iy - .5
					l = self.labels[i]
					plt.text(ix+1, iy+1, " & " + l)
					
				elif count[ix-1, iy-1] > 0:
    
					count[ix-1, iy-1] -= 1
					ix = ix - .5
					iy = iy - .5
					l = self.labels[i]
					plt.text(ix+1, iy+1, l)
		
		plt.show()

	def compute_centroids(self, heat, explicit=False):
		""" compute_centroids -- 計算map上的每一個中心
		
			parameters:
			- heat - 呈現heat map
			- explicit - 控制連結元件的形狀
			
			return value:
			- a list containing the matrices with the same x-y dims as the original map containing the centroid x-y coordinates
		"""

		xdim = self.xdim
		ydim = self.ydim
		centroid_x = np.matrix([[-1] * ydim for _ in range(xdim)])
		centroid_y = np.matrix([[-1] * ydim for _ in range(xdim)])

		heat = np.matrix(heat)

		def compute_centroid(ix, iy):
			# recursive function to find the centroid of a point on the map

			if (centroid_x[ix, iy] > -1) and (centroid_y[ix, iy] > -1):
				return {"bestx": centroid_x[ix, iy], "besty": centroid_y[ix, iy]}

			min_val = heat[ix, iy]
			min_x = ix
			min_y = iy

			# (ix, iy) is an inner map element
			if ix > 0 and ix < xdim-1 and iy > 0 and iy < ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is 左下角
			elif ix == 0 and iy == 0:

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

			# (ix, iy) is 右下角
			elif ix == xdim-1 and iy == 0:

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is 右上角
			elif ix == xdim-1 and iy == ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is 左上角
			elif ix == 0 and iy == ydim-1:

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

			# (ix, iy) is 左邊的元素
			elif ix == 0 and iy > 0 and iy < ydim-1:

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

			# (ix, iy) is 底邊的元素
			elif ix > 0 and ix < xdim-1 and iy == 0:

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy
	
				if heat[ix+1, iy+1] < min_val:
					min_val = heat[ix+1, iy+1]
					min_x = ix+1
					min_y = iy+1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is a 右邊的元素
			elif ix == xdim-1 and iy > 0 and iy < ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix, iy+1] < min_val:
					min_val = heat[ix, iy+1]
					min_x = ix
					min_y = iy+1

				if heat[ix-1, iy+1] < min_val:
					min_val = heat[ix-1, iy+1]
					min_x = ix-1
					min_y = iy+1

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

			# (ix, iy) is a 上面的元素
			elif ix > 0 and ix < xdim-1 and iy == ydim-1:

				if heat[ix-1, iy-1] < min_val:
					min_val = heat[ix-1, iy-1]
					min_x = ix-1
					min_y = iy-1

				if heat[ix, iy-1] < min_val:
					min_val = heat[ix, iy-1]
					min_x = ix
					min_y = iy-1

				if heat[ix+1, iy-1] < min_val:
					min_val = heat[ix+1, iy-1]
					min_x = ix+1
					min_y = iy-1

				if heat[ix+1, iy] < min_val:
					min_val = heat[ix+1, iy]
					min_x = ix+1
					min_y = iy

				if heat[ix-1, iy] < min_val:
					min_val = heat[ix-1, iy]
					min_x = ix-1
					min_y = iy

	        # if successful
	        # move to the square with the smaller value, i_e_, call
	        # compute_centroid on this new square
	        # note the RETURNED x-y coords in the centroid_x and
	        # centroid_y matrix at the current location
	        # return the RETURNED x-y coordinates

			if min_x != ix or min_y != iy:
				r_val = compute_centroid(min_x, min_y)

	            # if explicit is set show the exact connected component
	            # otherwise construct a connected componenent where all
	            # nodes are connected to a centrol node
				if explicit:

					centroid_x[ix, iy] = min_x
					centroid_y[ix, iy] = min_y
					return {"bestx": min_x, "besty": min_y}

				else:
					centroid_x[ix, iy] = r_val['bestx']
					centroid_y[ix, iy] = r_val['besty']
					return r_val

			else:
				centroid_x[ix, iy] = ix
				centroid_y[ix, iy] = iy
				return {"bestx": ix, "besty": iy}

		for i in range(xdim):
			for j in range(ydim):
				compute_centroid(i, j)

		return {"centroid_x": centroid_x, "centroid_y": centroid_y}

	def compute_combined_clusters(self, heat, explicit, rang):

		# compute the connected components
		centroids = self.compute_centroids(heat, explicit)
		# Get unique centroids
		unique_centroids = self.get_unique_centroids(centroids)
		# Get distance from centroid to cluster elements for all centroids
		within_cluster_dist = self.distance_from_centroids(centroids,
														   unique_centroids,
														   heat)
		# Get average pairwise distance between clusters
		between_cluster_dist = self.distance_between_clusters(centroids,
															  unique_centroids,	
															  heat)
		# Get a boolean matrix of whether two components should be combined
		combine_cluster_bools = self.combine_decision(within_cluster_dist,
													  between_cluster_dist,
													  rang)
		# Create the modified connected components grid
		ne_centroid = self.new_centroid(combine_cluster_bools,
										centroids,
										unique_centroids)

		return ne_centroid

	def get_unique_centroids(self, centroids):
		""" get_unique_centroids -- 取得唯一的中心 a function that computes a list of unique centroids from
		                            a matrix of centroid locations.
		
			parameters:
			- centroids - a matrix of the centroid locations in the map
		"""

		# get the dimensions of the map
		xdim = self.xdim
		ydim = self.ydim
		xlist = []
		ylist = []
		x_centroid = centroids['centroid_x']
		y_centroid = centroids['centroid_y']

		for ix in range(xdim):
			for iy in range(ydim):
				cx = x_centroid[ix, iy]
				cy = y_centroid[ix, iy]

		# Check if the x or y of the current centroid is not in the list
		# and if not
		# append both the x and y coordinates to the respective lists
				if not(cx in xlist) or not(cy in ylist):
					xlist.append(cx)
					ylist.append(cy)

		# return a list of unique centroid positions
		return {"position_x": xlist, "position_y": ylist}

	def distance_from_centroids(self, centroids, unique_centroids, heat):
		""" distance_from_centroids -- 用以取每一群集中心的平均距離的函數 

			parameters:
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
			- heat - a unified distance matrix
		"""

		centroids_x_positions = unique_centroids['position_x']
		centroids_y_positions = unique_centroids['position_y']
		within = []

		for i in range(len(centroids_x_positions)):
			cx = centroids_x_positions[i]
			cy = centroids_y_positions[i]

			# compute the average distance
			distance = self.cluster_spread(cx, cy, np.matrix(heat), centroids)

			# append the computed distance to the list of distances
			within.append(distance)

		return within

	def cluster_spread(self, x, y, umat, centroids):
		""" cluster_spread -- Function to calculate the average distance in
		                      one cluster given one centroid.
		
			parameters:
			- x - x position of a unique centroid
			- y - y position of a unique centroid
			- umat - a unified distance matrix
			- centroids - a matrix of the centroid locations in the map
		"""

		centroid_x = x
		centroid_y = y
		sum = 0
		elements = 0
		xdim = self.xdim
		ydim = self.ydim
		centroid_weight = umat[centroid_x, centroid_y]

		for xi in range(xdim):
			for yi in range(ydim):
				cx = centroids['centroid_x'][xi, yi]
				cy = centroids['centroid_y'][xi, yi]

				if(cx == centroid_x and cy == centroid_y):
					cweight = umat[xi, yi]
					sum = sum + abs(cweight - centroid_weight)
					elements = elements + 1

		average = sum / elements

		return average

	def distance_between_clusters(self, centroids, unique_centroids, umat):
		""" distance_between_clusters -- 群集間的平均距離
		
			parameters:
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
			- umat - a unified distance matrix
		"""

		cluster_elements = self.list_clusters(centroids, unique_centroids, umat)

		tmp_1 = np.zeros(shape=(max([len(cluster_elements[i]) for i in range(
				len(cluster_elements))]), len(cluster_elements)))

		for i in range(len(cluster_elements)):
			for j in range(len(cluster_elements[i])):
				tmp_1[j, i] = cluster_elements[i][j]

		columns = tmp_1.shape[1]

		tmp = np.matrix.transpose(np.array(list(combinations([i for i in range(columns)], 2))))

		tmp_3 = np.zeros(shape=(tmp_1.shape[0], tmp.shape[1]))

		for i in range(tmp.shape[1]):
			tmp_3[:, i] = np.where(tmp_1[:, tmp[1, i]]*tmp_1[:, tmp[0, i]] != 0,
									abs(tmp_1[:, tmp[0, i]]-tmp_1[:, tmp[1, i]]), 0)
	        # both are not equals 0

		mean = np.true_divide(tmp_3.sum(0), (tmp_3 != 0).sum(0))
		index = 0
		mat = np.zeros((columns, columns))

		for xi in range(columns-1):
			for yi in range(xi, columns-1):
				mat[xi, yi + 1] = mean[index]
				mat[yi + 1, xi] = mean[index]
				index = index + 1

		return mat

	def list_clusters(self, centroids, unique_centroids, umat):
		""" list_clusters -- A function to get the clusters as a list of lists.
		
			parameters:
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
			- umat - a unified distance matrix
		"""

		centroids_x_positions = unique_centroids['position_x']
		centroids_y_positions = unique_centroids['position_y']
		cluster_list = []

		for i in range(len(centroids_x_positions)):
			cx = centroids_x_positions[i]
			cy = centroids_y_positions[i]

	    # get the clusters associated with a unique centroid and store it in a list
			cluster_list.append(self.list_from_centroid(cx, cy, centroids, umat))

		return cluster_list

	def list_from_centroid(self, x, y, centroids, umat):
		""" list_from_centroid -- A funtion to get all cluster elements
		                          associated to one centroid.
		
			parameters:
			- x - the x position of a centroid
			- y - the y position of a centroid
			- centroids - a matrix of the centroid locations in the map
			- umat - a unified distance matrix
		"""

		centroid_x = x
		centroid_y = y
		xdim = self.xdim
		ydim = self.ydim

		cluster_list = []
		for xi in range(xdim):
			for yi in range(ydim):
				cx = centroids['centroid_x'][xi, yi]
				cy = centroids['centroid_y'][xi, yi]

				if(cx == centroid_x and cy == centroid_y):
					cweight = np.matrix(umat)[xi, yi]
					cluster_list.append(cweight)

		return cluster_list

	def combine_decision(self, within_cluster_dist, distance_between_clusters, rang):
		""" combine_decision -- 決定那些群集需要被合併
		
		
			parameters:
			- within_cluster_dist - A list of the distances from centroid to cluster elements for all centroids
			- distance_between_clusters - A list of the average pairwise distance between clusters
			- range - The distance where the clusters are merged together.
		"""

		inter_cluster = distance_between_clusters
		centroid_dist = within_cluster_dist
		dim = inter_cluster.shape[1]
		to_combine = np.matrix([[False]*dim]*dim)

		for xi in range(dim):
			for yi in range(dim):
				cdist = inter_cluster[xi, yi]
				if cdist != 0:
					rx = centroid_dist[xi] * rang
					ry = centroid_dist[yi] * rang
					if (cdist < centroid_dist[xi] + rx or
						cdist < centroid_dist[yi] + ry):
						to_combine[xi, yi] = True

		return to_combine

	def new_centroid(self, bmat, centroids, unique_centroids):
		""" new_centroid -- A function to combine centroids based on matrix of booleans.
		
			parameters:
			- bmat - a boolean matrix containing the centroids to merge
			- centroids - a matrix of the centroid locations in the map
			- unique_centroids - a list of unique centroid locations
		"""

		bmat_rows = bmat.shape[0]
		bmat_columns = bmat.shape[1]
		centroids_x = unique_centroids['position_x']
		centroids_y = unique_centroids['position_y']
		components = centroids

		for xi in range(bmat_rows):
			for yi in range(bmat_columns):
				if bmat[xi, yi]:
					x1 = centroids_x[xi]
					y1 = centroids_y[xi]
					x2 = centroids_x[yi]
					y2 = centroids_y[yi]
					components = self.swap_centroids(x1, y1, x2, y2, components)

		return components

	def swap_centroids(self, x1, y1, x2, y2, centroids):
		""" swap_centroids -- A function that changes every instance of a centroid to
		                      one that it should be combined with.
			parameters:
			- centroids - a matrix of the centroid locations in the map
		"""

		xdim = self.xdim
		ydim = self.ydim
		compn_x = centroids['centroid_x']
		compn_y = centroids['centroid_y']
		for xi in range(xdim):
			for yi in range(ydim):
				if compn_x[xi, 0] == x1 and compn_y[yi, 0] == y1:
					compn_x[xi, 0] = x2
					compn_y[yi, 0] = y2

		return {"centroid_x": compn_x, "centroid_y": compn_y}

	def embed(self, conf_int=.95, verb=False, ks=False):
		""" embed -- evaluate the embedding of a map using the F-test and
		             a Bayesian estimate of the variance in the training data.
		
			parameters:
			- conf_int - the confidence interval of the convergence test (default 95%)
			- verb - switch that governs the return value false: single convergence value
			  		 is returned, true: a vector of individual feature congences is returned.
			
			- return value:
			- return is the cembedding of the map (variance captured by the map so far)

			Hint: 
				  the embedding index is the variance of the training data captured by the map;
			      maps with convergence of less than 90% are typically not trustworthy.  Of course,
			      the precise cut-off depends on the noise level in your training data.
		"""

		if ks:
			return self.embed_ks(conf_int, verb)
		else:
			return self.embed_vm(conf_int, verb)

	def embed_ks(self, conf_int=0.95, verb=False):
		""" embed_ks -- using the kolgomorov-smirnov test """

		# map_df is a dataframe that contains the neurons
		map_df = self.neurons

		# data_df is a dataframe that contain the training data
		data_df = np.array(self.data)

		nfeatures = map_df.shape[1]

		# use the Kolmogorov-Smirnov Test to test whether the neurons and training
		# data appear
		# to come from the same distribution
		ks_vector = []
		for i in range(nfeatures):
			ks_vector.append(stats.mstats.ks_2samp(map_df[:, i], data_df[:, i]))

		prob_v = self.significance(graphics=False)
		var_sum = 0

		# compute the variance captured by the map
		for i in range(nfeatures):

			# the second entry contains the p-value
			if ks_vector[i][1] > (1 - conf_int):
				var_sum = var_sum + prob_v[i]
			else:
				# not converged - zero out the probability
				prob_v[i] = 0

		# return the variance captured by converged features
		if verb:
			return prob_v
		else:
			return var_sum

	def embed_vm(self, conf_int=.95, verb=False):
		""" embed_vm -- using variance and mean tests  """

		# map_df is a dataframe that contains the neurons
		map_df = self.neurons

		# data_df is a dataframe that contain the training data
		data_df = np.array(self.data)

		def df_var_test(df1, df2, conf=.95):

			if df1.shape[1] != df2.shape[1]:
				sys.exit("df_var_test: cannot compare variances of data frames")

			# init our working arrays
			var_ratio_v = [randint(1, 1) for _ in range(df1.shape[1])]
			var_confintlo_v = [randint(1, 1) for _ in range(df1.shape[1])]
			var_confinthi_v = [randint(1, 1) for _ in range(df1.shape[1])]

			def var_test(x, y, ratio=1, conf_level=0.95):

				DF_x = len(x) - 1
				DF_y = len(y) - 1
				V_x = stat.variance(x.tolist())
				V_y = stat.variance(y.tolist())

				ESTIMATE = V_x / V_y

				BETA = (1 - conf_level) / 2
				CINT = [ESTIMATE / f.ppf(1 - BETA, DF_x, DF_y),
						ESTIMATE / f.ppf(BETA, DF_x, DF_y)]

				return {"estimate": ESTIMATE, "conf_int": CINT}

		    # compute the F-test on each feature in our populations
			for i in range(df1.shape[1]):

				t = var_test(df1[:, i], df2[:, i], conf_level=conf)
				var_ratio_v[i] = t['estimate']
				var_confintlo_v[i] = t['conf_int'][0]
				var_confinthi_v[i] = t['conf_int'][1]

			# return a list with the ratios and conf intervals for each feature
			return {"ratio": var_ratio_v,
					"conf_int_lo": var_confintlo_v,
					"conf_int_hi": var_confinthi_v}

		def df_mean_test(df1, df2, conf=0.95):

			if df1.shape[1] != df2.shape[1]:
				sys.exit("df_mean_test: cannot compare means of data frames")

			# init our working arrays
			mean_diff_v = [randint(1, 1) for _ in range(df1.shape[1])]
			mean_confintlo_v = [randint(1, 1) for _ in range(df1.shape[1])]
			mean_confinthi_v = [randint(1, 1) for _ in range(df1.shape[1])]

			def t_test(x, y, conf_level=0.95):
				estimate_x = np.mean(x)
				estimate_y = np.mean(y)
				cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
				conf_int_lo = cm.tconfint_diff(alpha=1-conf_level, usevar='unequal')[0]
				conf_int_hi = cm.tconfint_diff(alpha=1-conf_level, usevar='unequal')[1]

				return {"estimate": [estimate_x, estimate_y],
						"conf_int": [conf_int_lo, conf_int_hi]}

			# compute the F-test on each feature in our populations
			for i in range(df1.shape[1]):
				t = t_test(x=df1[:, i], y=df2[:, i], conf_level=conf)
				mean_diff_v[i] = t['estimate'][0] - t['estimate'][1]
				mean_confintlo_v[i] = t['conf_int'][0]
				mean_confinthi_v[i] = t['conf_int'][1]

			# return a list with the ratios and conf intervals for each feature
			return {"diff": mean_diff_v,
					"conf_int_lo": mean_confintlo_v,
					"conf_int_hi": mean_confinthi_v}
		# do the F-test on a pair of datasets
		vl = df_var_test(map_df, data_df, conf_int)

		# do the t-test on a pair of datasets
		ml = df_mean_test(map_df, data_df, conf=conf_int)

		# compute the variance captured by the map --
		# but only if the means have converged as well.
		nfeatures = map_df.shape[1]
		prob_v = self.significance(graphics=False)
		var_sum = 0

		for i in range(nfeatures):

			if (vl['conf_int_lo'][i] <= 1.0 and vl['conf_int_hi'][i] >= 1.0 and
				ml['conf_int_lo'][i] <= 0.0 and ml['conf_int_hi'][i] >= 0.0):

				var_sum = var_sum + prob_v[i]
			else:
				# not converged - zero out the probability
				prob_v[i] = 0

		# return the variance captured by converged features
		if verb:
			return prob_v
		else:
			return var_sum

	def topo(self, k=50, conf_int=.95, verb=False, interval=True):
		""" topo -- measure the topographic accuracy of the map using sampling
		
			parameters:
			- k - the number of samples used for the accuracy computation
			- conf_int - the confidence interval of the accuracy test (default 95%)
			- verb - switch that governs the return value, false: single accuracy value
			  		 is returned, true: a vector of individual feature accuracies is returned.
			- interval - a switch that controls whether the confidence interval is computed.
			
			- return value is the estimated topographic accuracy
		"""
		

		# data.df is a matrix that contains the training data
		data_df = self.data

		if (k > data_df.shape[0]):
			sys.exit("topo: sample larger than training data.")

		data_sample_ix = [randint(1, data_df.shape[0]) for _ in range(k)]

		# compute the sum topographic accuracy - the accuracy of a single sample
		# is 1 if the best matching unit is a neighbor otherwise it is 0
		acc_v = []
		for i in range(k):
			acc_v.append(self.accuracy(data_df.iloc[data_sample_ix[i]-1], data_sample_ix[i]))

		# compute the confidence interval values using the bootstrap
		if interval:
			bval = self.bootstrap(conf_int, data_df, k, acc_v)

		# the sum topographic accuracy is scaled by the number of samples -
		# estimated
		# topographic accuracy
		if verb:
			return acc_v
		else:
			val = np.sum(acc_v)/k
			if interval:
				return {'val': val, 'lo': bval['lo'], 'hi': bval['hi']}
			else:
				return val

	def bootstrap(self, conf_int, data_df, k, sample_acc_v):
		""" bootstrap -- compute the topographic accuracies for the given confidence interval """

		ix = int(100 - conf_int*100)
		bn = 200

		bootstrap_acc_v = [np.sum(sample_acc_v)/k]

		for i in range(2, bn+1):

			bs_v = np.array([randint(1, k) for _ in range(k)])-1
			a = np.sum(list(np.array(sample_acc_v)[list(bs_v)]))/k
			bootstrap_acc_v.append(a)

		bootstrap_acc_sort_v = np.sort(bootstrap_acc_v)

		lo_val = bootstrap_acc_sort_v[ix-1]
		hi_val = bootstrap_acc_sort_v[bn-ix-1]

		return {'lo': lo_val, 'hi': hi_val}	

	def accuracy(self, sample, data_ix):
		""" accuracy -- the topographic accuracy of a single sample is 1 is the best matching unit
		             	and the second best matching unit are are neighbors otherwise it is 0
		"""

		o = self.best_match(sample, full=True)
		best_ix = o[0]
		second_best_ix = o[1]

		# sanity check
		coord = self.coordinate(best_ix)
		coord_x = coord[0]
		coord_y = coord[1]

		map_ix = self.visual[data_ix-1]
		coord = self.coordinate(map_ix)
		map_x = coord[0]
		map_y = coord[1]

		if (coord_x != map_x or coord_y != map_y or best_ix != map_ix):
			print("Error: best_ix: ", best_ix, " map_ix: ", map_ix, "\n")

		# determine if the best and second best are neighbors on the map
		best_xy = self.coordinate(best_ix)
		second_best_xy = self.coordinate(second_best_ix)
		diff_map = np.array(best_xy) - np.array(second_best_xy)
		diff_map_sq = diff_map * diff_map
		sum_map = np.sum(diff_map_sq)
		dist_map = np.sqrt(sum_map)

		# it is a neighbor if the distance on the map
		# between the bmu and 2bmu is less than 2,   should be 1 or 1.414
		if dist_map < 2:
			return 1
		else:
			return 0

	def best_match(self, obs, full=False):
		""" best_match -- given observation obs, return the best matching neuron """

	   	# NOTE: replicate obs so that there are nr rows of obs
		obs_m = np.tile(obs, (self.neurons.shape[0], 1))
		diff = self.neurons - obs_m
		squ = diff * diff
		s = np.sum(squ, axis=1)
		d = np.sqrt(s)
		o = np.argsort(d)

		if full:
			return o
		else:
			return o[0]

	def significance(self, graphics=True, feature_labels=False):
		""" significance -- compute the relative significance of each feature and plot it
		
			parameters:
			- graphics - a switch that controls whether a plot is generated or not
			- feature_labels - a switch to allow the plotting of feature names vs feature indices
			
			return value:
			- a vector containing the significance for each feature  
		"""

		data_df = self.data
		nfeatures = data_df.shape[1]

	    # Compute the variance of each feature on the map
		var_v = [randint(1, 1) for _ in range(nfeatures)]

		for i in range(nfeatures):
			var_v[i] = np.var(np.array(data_df)[:, i])

	    # we use the variance of a feature as likelihood of
	    # being an important feature, compute the Bayesian
	    # probability of significance using uniform priors

		var_sum = np.sum(var_v)
		prob_v = var_v/var_sum

	    # plot the significance
		if graphics:
			y = max(prob_v)

			plt.axis([0, nfeatures+1, 0, y])

			x = np.arange(1, nfeatures+1)
			tag = list(data_df)

			plt.xticks(x, tag)
			plt.yticks = np.linspace(0, y, 5)

			i = 1
			for xc in prob_v:
				plt.axvline(x=i, ymin=0, ymax=xc)
				i += 1

			plt.xlabel('Features')
			plt.ylabel('Significance')
			plt.show()
		else:
			return prob_v

	def projection(self):
		""" projection -- print the association of labels with map elements
			
			parameters:
			
			return values:
			- a dataframe containing the projection onto the map for each observation
		"""

		labels_v = self.labels
		x_v = []
		y_v = []

		for i in range(len(labels_v)):

			ix = self.visual[i]
			coord = self.coordinate(ix)
			x_v.append(coord[0])
			y_v.append(coord[1])

		return pd.DataFrame({'labels': labels_v, 'x': x_v, 'y': y_v})

	def neuron(self, x, y):
		""" neuron -- returns the contents of a neuron at (x,y) on the map as a vector
		
			parameters:
			 - x - map x-coordinate of neuron
			 - y - map y-coordinate of neuron
		
			return value:
			 - a vector representing the neuron
		"""

		ix = self.rowix(x, y)
		return self.neurons[ix]

	def coordinate(self, rowix):
		""" coordinate -- convert from a row index to a map xy-coordinate  """

		x = (rowix) % self.xdim
		y = (rowix) // self.xdim
		return [x, y]

	def rowix(self, x, y):
		""" rowix -- convert from a map xy-coordinate to a row index  """

		rix = x + y*self.xdim
		return rix

	def smooth_2d(self, Y, ind=None, weight_obj=None, grid=None, nrow=64, ncol=64, surface=True, theta=None):
		""" smooth_2d -- Kernel Smoother For Irregular 2-D Data """

		def exp_cov(x1, x2, theta=2, p=2, distMat=0):
			x1 = x1*(1/theta)
			x2 = x2*(1/theta)
			distMat = euclidean_distances(x1, x2)
			distMat = distMat**p
			return np.exp(-distMat)

		NN = [[1]*ncol] * nrow
		grid = {'x': [i for i in range(nrow)], "y": [i for i in range(ncol)]}

		if weight_obj is None:
			dx = grid['x'][1] - grid['x'][0]
			dy = grid['y'][1] - grid['y'][0]
			m = len(grid['x'])
			n = len(grid['y'])
			M = 2 * m
			N = 2 * n
			xg = []

			for i in range(N):
				for j in range(M):
					xg.extend([[j, i]])

			xg = np.matrix(xg)

			center = []
			center.append([int(dx * M/2-1), int((dy * N)/2-1)])

			out = exp_cov(xg, np.matrix(center),theta=theta)
			out = np.matrix.transpose(np.reshape(out, (N, M)))
			temp = np.zeros((M, N))
			temp[int(M/2-1)][int(N/2-1)] = 1

			wght = np.fft.fft2(out)/(np.fft.fft2(temp) * M * N)
			weight_obj = {"m": m, "n": n, "N": N, "M": M, "wght": wght}

		temp = np.zeros((weight_obj['M'], weight_obj['N']))
		temp[0:m, 0:n] = Y
		temp2 = np.fft.ifft2(np.fft.fft2(temp) *
							 weight_obj['wght']).real[0:weight_obj['m'],
													  0:weight_obj['n']]

		temp = np.zeros((weight_obj['M'], weight_obj['N']))
		temp[0:m, 0:n] = NN
		temp3 = np.fft.ifft2(np.fft.fft2(temp) *
							 weight_obj['wght']).real[0:weight_obj['m'],
													  0:weight_obj['n']]

		return temp2/temp3
#初始化模型 #xdim, ydim 皆要大於3 #見74行
m = som.map(xdim=10,ydim=5)

#訓練資料
m.fit(attr,animal)

#呈現分群
m.starburst()
