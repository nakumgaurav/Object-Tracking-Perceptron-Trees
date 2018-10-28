# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader
import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# parser = argparse.ArgumentParser()
# parser.add_argument("epochs", default=5)
# args = parser.parse_args()
n_epoch = 5


threshold = 0.0001

class Perceptron():
	def __init__(self, dataset):
		self.dataset = dataset


	# Make a prediction with weights
	@staticmethod
	def predict(row, weights):
		activation = weights[0]
		for i in range(len(row)-1):
			activation += weights[i + 1] * row[i]
		return 1.0 if activation >= 0.0 else 0.0

	# Estimate Perceptron weights using stochastic gradient descent
	def train_weights(self, train, l_rate, n_epoch):
		weights = [0.0 for i in range(len(train[0]))]
		for epoch in range(n_epoch):
			for row in train:
				prediction = Perceptron.predict(row, weights)
				error = row[-1] - prediction
				weights[0] = weights[0] + l_rate * error
				for i in range(len(row)-1):
					weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		return weights

	# Perceptron Algorithm With Stochastic Gradient Descent
	def perceptron(train, test, l_rate, n_epoch):
		predictions = list()
		weights = train_weights(train, l_rate, n_epoch)
		for row in test:
			prediction = predict(row, weights)
			predictions.append(prediction)
		return(predictions)

	def driver(self):
		seed(1)

		# evaluate algorithm
		# n_folds = 3
		l_rate = 0.01
		# try:
		n_epoch = 5#int(args.epochs)
		# except:
		# 	n_epoch = 10000
		# Prepare the training data
		train = self.dataset
		weights = self.train_weights(train, l_rate, n_epoch)
		return weights
		# scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
		# print('Scores: %s' % scores)
		# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


def gini_impurity(data):
	num_instances = float(len(data))

	np_data = np.array(data)
	num_ones = sum(np_data[:,-1])*1.0
	p_1 = num_ones/num_instances
	p_0 = 1.0 - p_1

	gini = 1.0 - (p_0**2 + p_1**2)
	return gini


# performs k-means clustering on the data and returns clusters
def k_means(data):
	num_points = data.shape[0]
	K = int(0.2 * num_points)
	if(K==0):
		K=1
	# apply K-means clustering
	kmeans = KMeans(K).fit(data)
	return kmeans

# takes the data points of a particular cluster and inflates it
# returns the new data points to be added to the original dataset
# num_new_points represents the number of points to be had
# in the inflated cluster
def inflate_cluster(cluster, num_new_points):
	# print "Inflating..."
	# find the label of this cluster
	y = cluster[0,-1]
	# remove the label column of the data
	cluster = cluster[:,:-1]
	# find the number of new points to be added
	num_old_points = cluster.shape[0]
	num_new_points -= num_old_points

	# if(num_new_points <= 0):
	# 	new_points = np.array((0,cluster.shape[1]))
	# 	return new_points, 0

	# initialize the new_ponts matrix

	# print "num_old_points"
	# print num_old_points
	# print "num_new_points"
	# print num_new_points
	new_points = np.zeros((num_new_points,cluster.shape[1]))
	a = 0
	b = num_old_points

	m = min(10,b/2+1)

	# print("m={0} num_new_points={1}".format(m,num_new_points))

	neigh = NearestNeighbors(n_neighbors=m)
	# print cluster.shape
	neigh.fit(cluster)

	for i in xrange(num_new_points):
		# x_l is selected randomly from the cluster
		x_l = cluster[np.random.randint(a,b)]

		# x_j is selected randomly from the m-nearest neighbors of x_l
		nearest_ind = neigh.kneighbors(x_l.reshape(1,x_l.size), return_distance=False)
		rand_ind = np.random.randint(0,m)
		# get one of the m nearest points chosen randomly
		x_j = cluster[nearest_ind[0,rand_ind]]

		# choose alpha randomly from (0,1)
		alpha = np.random.random()

		# construct x_new
		x_new = x_l + alpha*(x_j - x_l)
		new_points[i] = x_new

	Y = np.ones((num_new_points,1))*y
	new_points = np.concatenate((new_points,Y),axis=1)
	# print "Done Inflating"
	return new_points


# divide the data into majority and minority sets
def get_maj_min(data):
	# print data.shape
	num_zeros = sum(data[:,-1] == 0)
	num_ones = sum(data[:,-1] == 1)


	# print "In get_maj_min..."
	# print num_zeros
	# print num_ones

	neg_data = data[data[:,-1]==0]
	pos_data = data[data[:,-1]==1]

	k_means_pos = k_means(pos_data[:,:-1])
	k_means_neg = k_means(neg_data[:,:-1])

	k_means_maj = k_means_neg
	k_means_min = k_means_pos
	maj_data = neg_data
	min_data = pos_data
	# Positive set is the majority class
	if(num_ones > num_zeros):
		k_means_maj = k_means_pos
		k_means_min = k_means_neg
		maj_data = pos_data
		min_data = neg_data

	return k_means_maj, k_means_min, maj_data, min_data

# given unbalanced data, balances the data and returns the same
def balance_data(data):
	k_means_maj, k_means_min, maj_data, min_data = get_maj_min(data)

	# get the Kmeans labels for the data points
	labels = np.array(k_means_maj.labels_)
	
	# count the number of occurences of each label (cluster number)
	counts = np.bincount(labels)


	# index of cluster with maximum number of points
	Cmaj_ind = np.argmax(counts)
	# number of points in the cluster of maximum size (this is the Cmaj of paper)
	Cmaj_num = counts[Cmaj_ind] # which is same as sum(labels==Cmax_ind)

	# inflate the clusters of the majority set
	# first, find the cluster centres
	k_means_maj_clusters = k_means_maj.cluster_centers_
	Kmaj = k_means_maj_clusters.shape[0] # which is same as len(list(set(labels)))

	new_maj_data = maj_data
	# inflate each cluster
	for i in xrange(Kmaj):
		# print "Cluster # %d" %i
		# skip the biggest clusters
		if(counts[i]==Cmaj_num):
			continue

		# print (labels==i).size
		# print maj_data.shape

		# find the data points which belong to cluster index i
		Ci = maj_data[labels==i]

		if(counts[i]==0):
			print("Maj Labels:")
			print(labels)
			print("Maj Counts:")
			print(counts)

		# inflate the cluster i by finding the new points
		new_points = inflate_cluster(Ci,Cmaj_num)
		# add the new points to the maj_data
		new_maj_data = np.concatenate((new_maj_data,new_points))

	# inflate the clusters of the minority set
	Nmaj = Cmaj_num * Kmaj

	min_labels = k_means_min.labels_
	min_counts = np.bincount(min_labels)

	k_means_min_clusters = k_means_min.cluster_centers_
	Kmin = k_means_min_clusters.shape[0]
	print("Kmin Check")
	print(Kmin)
	# print k_means_min_clusters
	# number of data points to be had in each inflated minority cluster

	if(min_counts.size < Kmin):
		Kmin = min_counts.size
		print("counts.size ERROR")
		print(min_labels)
		print(min_counts)
		print("My foot")

	Knew = int(Nmaj/Kmin)
	
	print("Knew check")
	print(Knew)

	new_min_data = min_data
	for i in xrange(Kmin):
		if(min_counts[i]>=Knew):
			continue

		# if(min_counts[i]==0):
		# 	print "MinCounts[i]=0 ERROR"
		# 	print min_labels
		# 	print min_counts

		# find the data points which belong to cluster index i
		Ci = min_data[min_labels==i]

		if(Ci.size==0):
			print("MinCounts[i]=0 ERROR")
			print(min_labels)
			print(min_counts)

		# inflate the cluster i by finding the new points
		new_points = inflate_cluster(Ci,Knew)
		# add the new points to the maj_data
		new_min_data = np.concatenate((new_min_data,new_points))

	# merge the inflated data clusters to get the new balanced dataset
	# print Nmaj
	# print Kmin
	# print new_maj_data.shape
	# print new_min_data.shape
	inflated_data = np.concatenate((new_maj_data, new_min_data))
	np.random.shuffle(inflated_data)
	return inflated_data


def get_groups(obj, train):
	# print "Training Perceptron..."
	weights = obj.driver()
	# print "Perceptron Trained"

	left = list()
	right = list()

	for row in train:
		prediction = Perceptron.predict(row, weights)
		if(prediction==0.0):
			left.append(row)
		else:
			right.append(row)

	left = np.array(left)
	right = np.array(right)
	return left, right, weights

def get_split(train):
	node  = {}
	impurity = gini_impurity(train)
	print(impurity)
	if(impurity<threshold):
		print("Leaf finally!")
		left = right = to_terminal(train)
	else:
		train = balance_data(train)
		obj = Perceptron(train)
		left, right, weights = get_groups(obj, train)
		print("Perceptron says...")
		print(len(left), len(right))
		node['W'] = weights

	# print("Left={}".format(left))
	# print("Right={}".format(right))

	node['left'] = left
	node['right'] = right
	return node

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	label =  max(set(outcomes), key=outcomes.count)
	label_arr = np.array(outcomes)
	num_samps = len(outcomes)
	num_pos = sum(label_arr==1)
	return label, num_pos, num_samps
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	print("Depth:")
	print(depth)
	left, right = node['left'], node['right']
	# check for a no split
	if left is None or (type(left)==np.ndarray and left.size==0) or right is None or (type(right)==np.ndarray and right.size==0):
		node['left'] = node['right'] = to_terminal(left + right)
		print("CORRECT1")
		return
	# check for max depth
	if depth >= max_depth:
		print("CORRECT2")
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if(type(left)==int):
		node['left'] = left
	elif len(left) <= min_size:
		print("CORRECT3l")
		node['left'] = to_terminal(left)
	else:
		# if the entire node has the same predictions, make it a terminal node
		if(sum(left[:,-1])==0 or sum(left[:,-1])==left.shape[0]):
			print("CORRECT4l")
			node['left'] = to_terminal(left)
		else:
			# balance the data
			# left = balance_data(left)
			# print "Balancing says..."
			# print_num_zeros_ones(left)
			# obj = Perceptron(left)
			node['left'] = get_split(left)
			split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if(type(right)==int):
		node['right'] = right
	elif len(right) <= min_size:
		print("CORRECT3r")
		node['right'] = to_terminal(right)
	else:
		# print len(right)
		if(sum(right[:,-1])==0 or sum(right[:,-1])==right.shape[0]):
			print("CORRECT4r")
			node['right'] = to_terminal(right)
		else:
			# balance the data
			# right = balance_data(right)
			# print "Balancing says..."
			# print_num_zeros_ones(right)
			# obj = Perceptron(right)
			node['right'] = get_split(right)
			# print len(node['right']['left']), len(node['right']['right'])
			split(node['right'], max_depth, min_size, depth+1)

# prints the number of zeros and ones in the data
def print_num_zeros_ones(data):
	print("Zeros")
	print(sum(data[:,-1]==0))
	print("Ones")
	print(sum(data[:,-1]==1))

def build_tree(dataset, max_depth=5, min_size=10):
	# Perceptron Tree
	# balance the initial data to the root node
	# dataset = balance_data(dataset)
	print_num_zeros_ones(dataset)
	# obj = Perceptron(dataset)
	root = get_split(dataset)
	# print root
	# print eval_algo(dataset, root)
	split(root, max_depth, min_size, 1)
	return root

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def prediction(row, root):
	if(type(root) == dict and 'W' not in root):
		print("WEIRD")
		return root['left']

	# leaf node
	if(type(root)==tuple):
		# if(type(root)==int or type(root)==np.float64):
		return root

	# print type(root)
	# print root

	weights = root['W']
	pred_val = Perceptron.predict(row, root['W'])
	# return pred_val
	if(pred_val==0):
		# if(type(root['left'])==int or type(root['left'])==np.float64):
		if(type(root['left']==tuple)):
			return root['left']
		# elif(type(root['left']) is np.ndarray):
		else:
			root = root['left']
			return prediction(row, root)
	if(pred_val==1):
		# if(type(root['right'])==int or type(root['right'])==np.float64):
		if(type(root['right']==tuple)):
			return root['right']
		# elif(type(root['right']) is np.ndarray):
		else:
			root = root['right']
			return prediction(row, root)

def eval_algo(data, root):
	num_correct = 0
	for row in data:
		pred_val, _, _ = prediction(row,root)
		num_correct += (pred_val==row[-1])
	return (num_correct*100.0)/len(data)*1.0


def main():
		# Test the Perceptron algorithm on the sonar dataset
		# load and prepare data
		# filename = 'sonar.all-data.csv'
		filename = 'data_banknote_authentication.csv'
		dataset = load_csv(filename)
		for i in range(len(dataset[0])-1):
			str_column_to_float(dataset, i)
		# convert string class to integers
		str_column_to_int(dataset, len(dataset[0])-1)

		# train = [[2.771244718,1.784783929,0], [1.728571309,1.169761413,0], [3.678319846,2.81281357,0], [3.961043357,2.61995032,0],
		# 	     [7.497545867,3.162953546,1], [9.00220326,3.339047188,1], [7.444542326,0.476683375,1], [10.12493903,3.234550982,1]]

		# test = [[6.642287351,3.319983761,1], [2.999208922,2.209014212,0]]

		# train = np.array(dataset[:80] + dataset[101:181], dtype=np.float)
		train = np.array(dataset[:540] + dataset[760:1300], dtype=np.float)
		# train = dataset[:193]
		# test = np.array(dataset[80:96] + dataset[181:197], dtype=np.float)
		test = np.array(dataset[501:600] + dataset[1301:1350], dtype=np.float)

		tree = build_tree(train)
		# print tree

		print("Train accuracy={}".format(eval_algo(train,tree)))
		print("Test accuracy={}".format(eval_algo(test,tree)))

		# from sklearn.tree import DecisionTreeClassifier
		# clf = DecisionTreeClassifier()
		# clf = clf.fit(train[:,:-1],train[:,-1])

		# y_pred = clf.predict(test[:,:-1])
		# # print y_pred
		# # print test[:,-1]
		# print sum(y_pred==test[:,-1])*100.0/test[:,-1].size*1.0


		# from sklearn.svm import SVC
		# clf = SVC(gamma='auto')
		# clf.fit(train[:,:-1],train[:,-1])

		# y_pred = clf.predict(test[:,:-1])
		# # print sum(y_pred==test[:,-1])*100.0/test[:,-1].size*1.0
		# from sklearn.metrics import classification_report, confusion_matrix  
		# print(confusion_matrix(test[:,-1],y_pred))


		# print Perceptron.predict(test[0], tree['W'])
		# print Perceptron.predict(test[1], tree['W'])

		# print tree


# # Split a dataset into k folds
# def cross_validation_split(dataset, n_folds):
# 	dataset_split = list()
# 	dataset_copy = list(dataset)
# 	fold_size = int(len(dataset) / n_folds)
# 	for i in range(n_folds):
# 		fold = list()
# 		while len(fold) < fold_size:
# 			index = randrange(len(dataset_copy))
# 			fold.append(dataset_copy.pop(index))
# 		dataset_split.append(fold)
# 	return dataset_split

# # Calculate accuracy percentage
# def accuracy_metric(actual, predicted):
# 	correct = 0
# 	for i in range(len(actual)):
# 		if actual[i] == predicted[i]:
# 			correct += 1
# 	return correct / float(len(actual)) * 100.0

# # Evaluate an algorithm using a cross validation split
# def evaluate_algorithm(dataset, algorithm, n_folds, *args):
# 	folds = cross_validation_split(dataset, n_folds)
# 	scores = list()
# 	for fold in folds:
# 		train_set = list(folds)
# 		train_set.remove(fold)
# 		train_set = sum(train_set, [])
# 		test_set = list()
# 		for row in fold:
# 			row_copy = list(row)
# 			test_set.append(row_copy)
# 			row_copy[-1] = None
# 		predicted = algorithm(train_set, test_set, *args)
# 		actual = [row[-1] for row in fold]
# 		accuracy = accuracy_metric(actual, predicted)
# 		scores.append(accuracy)
# 	return scores

if __name__ == "__main__":
	main()