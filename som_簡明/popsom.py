"""
#som 動物圖:註解篇
-環境適用:nntest3
-參考資料:https://github.com/njali2001/popsom
-pandas 0.20.3
-py版本 3.5~3.6
-搭配mainsom.py使用
# modify by jessica cheng & chu-chu lin 
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns					
from random import randint
from sklearn.metrics.pairwise import euclidean_distances
import statsmodels.stats.api as sms     # t-test
import statistics as stat               # F-test
from scipy import stats                 # KS Test
from scipy.stats import f               # F-test
from itertools import combinations

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

# 		
# 	def marginal(self, marginal):
# 		""" marginal -- 印出顯示神經元與資料的邊界機率分布

# 	 	 	parameters:
# 	 	 	- marginal:訓練資料的維度或索引即為marginal
# 	 	"""
		
# 		# check if the second argument is of type character
# 		if type(marginal) == str and marginal in list(self.data):

# 			f_ind = list(self.data).index(marginal)
# 			f_name = marginal
# 			train = np.matrix(self.data)[:, f_ind]
# 			neurons = self.neurons[:, f_ind]
# 			plt.ylabel('Density')
# 			plt.xlabel(f_name)
# 			sns.kdeplot(np.ravel(train),
# 				        label="training data",
# 						shade=True,
# 						color="b")
# 			sns.kdeplot(neurons, label="neurons", shade=True, color="r")
# 			plt.legend(fontsize=15)
# 			plt.show()

# 		elif (type(marginal) == int and marginal < len(list(self.data)) and marginal >= 0):

# 			f_ind = marginal
# 			f_name = list(self.data)[marginal]
# 			train = np.matrix(self.data)[:, f_ind]
# 			neurons = self.neurons[:, f_ind]
# 			plt.ylabel('Density')
# 			plt.xlabel(f_name)
# 			sns.kdeplot(np.ravel(train),
# 						label="training data",
# 						shade=True,
# 						color="b")
# 			sns.kdeplot(neurons, label="neurons", shade=True, color="r")
# 			plt.legend(fontsize=15)
# 			plt.show()

# 		else:
# 			sys.exit("marginal: second argument is not the name of a training \
# 						data frame dimension or index")

	def vsom_p(self):
		""" vsom_p -- 向量的，隨機som未優化的版本
		
    	"""
    	# 定義參數
		dr = self.data.shape[0]
		dc = self.data.shape[1]
		nr = self.xdim*self.ydim
		nc = dc  # 資料維度與神經元數量是相同的

	   # 建立與初始化神經元
		cells = nr * nc  # 神經元數量 乘上 資料維度

	    # 具有神經元初始值的向量
		v = np.random.uniform(-1, 1, cells)

	    # 每行代表神經元，每列代表維度
		neurons = np.transpose(np.reshape(v, (nc, nr)))  # 重新將向量排成矩陣

	    # 計算初始nsize(鄰居大小)和每一步驟(nsize_step)
		nsize = max(self.xdim, self.ydim) + 1
		nsize_step = np.ceil(self.train/nsize)
		step_counter = 0  # 計算每一個step的epoch數量

	    # 轉換一維列座標到二維圖 
		def coord2D(rowix):

			x = np.array(rowix) % self.xdim
			y = np.array(rowix) // self.xdim

			return np.concatenate((x, y))

	    # gamma函數的常數 #gamma函數的用途:更新權重與鄰近範圍
		m = [i for i in range(nr)]  # 這個向量有神經元的一維位址 #nr=xdim*ydim

	    # x-y coordinate of ith neuron: m2Ds[i,] = c(xi, yi)
		m2Ds = np.matrix.transpose(coord2D(m).reshape(2, nr))

	    # neighborhood function 鄰近函數
		def Gamma(c):

	        # 查看c的2d座標 
			c2D = m2Ds[c, ]
	        # 具有跟c2D相同行(row)的矩陣 
			c2Ds = np.outer(np.linspace(1, 1, nr), c2D)
	        # 每個神經元和c的距離向量，以map的座標呈現 
			d = np.sqrt(np.dot((c2Ds - m2Ds)**2, [1, 1]))
	        # 如果m 在圖上是鄰居，則 alpha，否則 0.0
			hood = np.where(d < nsize*1.5, self.alpha, 0.0)

			return hood
		
	    # training #訓練
		
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

	        # 競爭過程
			xk_m = np.outer(np.linspace(1, 1, nr), xk)  #nr=xdim*ydim
			#相差、平方、開根號-->歐氏距離
			#計算神經元與cluster中心的距離diff
			diff = neurons - xk_m
			# 取平方
			squ = diff * diff
			s = np.dot(squ, np.linspace(1, 1, nc))
			o = np.argsort(s)
			c = o[0]

	        # 更新:神經元權重與範圍
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
			return {"embed": embed, "topo": topo_}
		else:
			return (0.5*embed + 0.5*topo_)		

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
			# 轉換內部節點並計算umat值
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

		# 計算 umat values for corners #映射的方式有點奇妙，是反的
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
		ax.pcolor(tmp_1, cmap=plt.cm.summer) #這裡可以換底圖顏色
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

		# # # 將連結線放在map上
		# if comp:
		# 	if not merge:
		# 		# find the centroid for each neuron on the map
		# 		centroids = self.compute_centroids(heat, explicit)pyt
		# 	else:
		# 		# find the unique centroids for the neurons on the map
		# 		centroids = self.compute_combined_clusters(umat, explicit, merge_range)

		# 	# connect each neuron to its centroid
		# 	for ix in range(x):
		# 		for iy in range(y):
		# 			cx = centroids['centroid_x'][ix, iy]
		# 			cy = centroids['centroid_y'][ix, iy]
		# 			plt.plot([ix+0.5, cx+0.5],
	    #                      [iy+0.5, cy+0.5],
	    #                      color='grey',
	    #                      linestyle='-',
	    #                      linewidth=1.0)

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

				# label印出，如果有兩個label在同一點上，用","處理
				if count[ix-1, iy-1] > 1:

					count[ix-1, iy-1] -= 1
					ix = ix - .9
					iy = iy - .5
					l = self.labels[i]
					plt.text(ix+1, iy+1, "       ," + l)
					#plt.text(ix+1, iy+1, " & " + l)
					
				elif count[ix-1, iy-1] > 0:
				
					count[ix-1, iy-1] -= 1
					ix = ix - .9
					iy = iy - .5
					l = self.labels[i]
					plt.text(ix+1, iy+1, l)

		plt.show()

	def best_match(self, obs, full=False):
		""" best_match -- 給定觀測值, 回傳最符合的神經元 """

	   	# NOTE: 複製觀察值,使觀察值的row數符合nr行  #nr=xdim*ydim
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

	def coordinate(self, rowix):
		""" coordinate座標 -- 從行座標轉換成xy座標  """

		x = (rowix) % self.xdim
		y = (rowix) // self.xdim
		return [x, y]

	def smooth_2d(self, Y, ind=None, weight_obj=None, grid=None, nrow=64, ncol=64, surface=True, theta=None):
		""" 使不正常的2d影像平滑 """

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