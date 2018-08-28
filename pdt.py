# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("epochs")
args = parser.parse_args()

threshold = 0.01

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
		n_epoch = int(args.epochs)
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

def get_groups(obj, train):
	weights = obj.driver()
	left = list()
	right = list()
	for row in train:
		prediction = Perceptron.predict(row, weights)
		if(prediction==0.0):
			left.append(row)
		else:
			right.append(row)
	return left, right, weights

def get_split(obj, train):
	node  = {}
	if(gini_impurity(train)<threshold):
		left = right = to_terminal(train)
	else:	
		left, right, weights = get_groups(obj, train)
		print len(left), len(right)
		node['W'] = weights

	# print("Left={}".format(left))
	# print("Right={}".format(right))

	node['left'] = left 
	node['right'] = right
	return node

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['left'], node['right']
	# check for a no split
	if left is None or (type(left)==list and len(left)==0) or right is None or (type(right)==list and len(right)==0):
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if(type(left)==int):
		node['left'] = left
	elif len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		# print len(left)
		obj = Perceptron(left)
		node['left'] = get_split(obj, left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if(type(right)==int):
		node['right'] = right
	elif len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		# print len(right)	
		obj = Perceptron(right)
		node['right'] = get_split(obj, right)
		# print len(node['right']['left']), len(node['right']['right'])
		split(node['right'], max_depth, min_size, depth+1)

def build_tree(dataset, max_depth=10, min_size=10):
	# Perceptron Tree
	obj = Perceptron(dataset)
	root = get_split(obj,dataset)
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
	if('W' not in root):
		return root['left']

	pred_val = Perceptron.predict(row, root['W'])
	if(pred_val==0):
		if(type(root['left'])==int):
			return root['left']
		else:
			root = root['left']
			return prediction(row, root)
	if(pred_val==1):
		if(type(root['right'])==int):
			return root['right']
		else:
			root = root['right']
			return prediction(row, root)

def eval_algo(data, root):
	num_correct = 0
	for row in data:
		pred_val = prediction(row,root)
		num_correct += (pred_val==row[-1])
	return (num_correct*100.0)/len(data)*1.0


def main():
		# Test the Perceptron algorithm on the sonar dataset
		# load and prepare data
		filename = 'sonar.all-data.csv'
		dataset = load_csv(filename)
		for i in range(len(dataset[0])-1):
			str_column_to_float(dataset, i)
		# convert string class to integers
		str_column_to_int(dataset, len(dataset[0])-1)

		# train = [[2.771244718,1.784783929,0], [1.728571309,1.169761413,0], [3.678319846,2.81281357,0], [3.961043357,2.61995032,0],
		# 	     [7.497545867,3.162953546,1], [9.00220326,3.339047188,1], [7.444542326,0.476683375,1], [10.12493903,3.234550982,1]]

		# test = [[6.642287351,3.319983761,1], [2.999208922,2.209014212,0]]

		train = dataset[:80] + dataset[101:180]
		# train = dataset[:193]
		test = dataset[80:96] + dataset[180:196]

		tree = build_tree(train)
		print("Train accuracy={}".format(eval_algo(train,tree)))
		print("Test accuracy={}".format(eval_algo(test,tree)))
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
