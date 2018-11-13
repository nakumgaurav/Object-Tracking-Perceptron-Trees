import os
import numpy as np
import cv2
import skimage
import argparse
from skimage.feature import local_binary_pattern, hog, haar_like_feature
from skimage.transform import integral_image
from pdt import build_tree, prediction
from sklearn.linear_model import PassiveAggressiveClassifier as clfPA
from tqdm import tqdm

EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--frames_dir", default='new_frames/bag')
# parser.add_argument('-g', "--ground_truth_dir", default='gt/bag')
parser.add_argument('-o', "--output_dir", default='test/output')
args = parser.parse_args()

frames_dir = args.frames_dir
# ground_truth_dir = args.ground_truth_dir
output_dir = args.output_dir

forest_update = 10
classifier_update = 5

# Given the location of the object in the previous frame, find all patches 
# in the s-neighborhood of it and compute their feature vectors
# img -> current frame (frame at time t)
# loc_object -> location of object in frame (t-1)
# s -> neighborhood radius
# it is assumed that location is specified as location of top left corner
def get_patches(img,loc_object,s=30, patch_size=16):
	# The set X_s stores the locations of the desired patches
	X_s = list()
	# patch_features stores the feature vectors corr to each patch
	patch_features = list()

	# coordinates of the object patch center
	x_obj, y_obj = loc_object

	d = int(patch_size/2)
	# extract patches and compute their features using sliding window approach
	for x in range(x_obj-s,x_obj+s):
		for y in range(y_obj-s,y_obj+s):
			patch = img[x-d:x+d, y-d:y+d, :]
			X_s.append((x,y))
			features = compute_features(patch)
			patch_features.append(features)

	X_s = np.array(X_s)
	patch_features = np.array(patch_features)

	return X_s, patch_features


# compute four types of features for a patch (described within)
def compute_features(patch, cell_size1=4, cell_size2=3):
	# print("In compute_features:")

	patch_gray = patch.copy()
	patch_gray = cv2.cvtColor(patch_gray, cv2.COLOR_BGR2GRAY)


	# refer RTCT paper
	def random_projection_mat(m, n=100):
		# Approach -1:
		# R = np.random.normal(0,1,(n,m))

		# Approach-2:
		s = 3
		s *= 1.0
		R = np.random.uniform(-s,s, (n,m))
		R[R<-s] = -1
		mask1 = R<s
		mask2 = R>=-s
		mask = mask1 & mask2
		R[mask] = 0
		R[R>=s] = 1

		return R

	
	patch_size = patch.shape[0]

	# print("Feature-1: Intensity Histogram")
	## Feature-1: Intensity Histogram
	intensity_hist = np.empty((0,))
	num_cells1 = int(patch_size/cell_size1)
	num_pixels_cell1 = cell_size1*cell_size1

	for i in range(num_cells1):
		for j in range(num_cells1):
			cell = patch_gray[i*cell_size1:(i+1)*cell_size1, j*cell_size1:(j+1)*cell_size1]
			hist = np.histogram(cell, bins=num_pixels_cell1, range=(0.0,255.0), density=True) # Normalized histogram
			# np.histogram returns a 2-tuple, where 1st element represent the y-values of each bin, 2nd element
			# represents the x-values of each bin
			hist = hist[0] 
			# print("HEEEEREEE", hist.shape)
			# print(hist)
			intensity_hist = np.concatenate((intensity_hist, hist))

	intensity_hist = intensity_hist/(np.sum(intensity_hist) + EPS)
	# print(intensity_hist.shape)

	# print("Feature-2: LBP")
	## Feature-2: Local Binary Pattern (grayscale and roation invariant)
	radius = 1
	num_points = radius * 8
	lbp_hist = np.empty((0,))

	for i in range(num_cells1):
		for j in range(num_cells1):
			cell = patch_gray[i*cell_size1:(i+1)*cell_size1, j*cell_size1:(j+1)*cell_size1]
			lbp_cell = skimage.feature.local_binary_pattern(cell, num_points, radius, method='uniform')
			hist = np.histogram(cell, bins=num_pixels_cell1, range=(0.0,255.0), density=True) # Normalized histogram
			hist = hist[0]
			lbp_hist = np.concatenate((lbp_hist, hist))
			
	lbp_hist = lbp_hist/(np.sum(lbp_hist) + EPS)
	# print(lbp_hist.shape)

	# print("Feature-3: HOG")
	## Feature-3: HOG (without spatial block-division modes)
	hog_hist = skimage.feature.hog(patch_gray, orientations=9, pixels_per_cell=(cell_size2,cell_size2), cells_per_block=(1,1), block_norm='L1')
	
	hog_hist = hog_hist/(np.sum(hog_hist) + EPS)
	# print(hog_hist)
	# print(hog_hist.shape)

	# print("Feature-4: Haar")
	## Feature-4: Haar-like features
	## One approach to filter relevant features is to train a random forest classifier on all 32176 features computed for a 16x16 patch (skimage face dec docs)
	## Another approach is to reduce the dimensions of the haar feature vector using random projection (real-time compressive tracking)
	ii = integral_image(patch_gray)
	haar_vec = skimage.feature.haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1])
	R = random_projection_mat(haar_vec.size)
	haar_feat = np.matmul(R,haar_vec)

	haar_feat = haar_feat/(np.sum(haar_feat) + EPS)
	# print(haar_feat.shape)

	## Concatenate all features
	return np.concatenate((intensity_hist, lbp_hist, hog_hist, haar_feat))


# generate a random sample having equal number of positive and negative samples
# total F samples are generated
# by convention last column of the data matrix consists of the labels
def random_sample(train, F=100):
	train_data = np.copy(train)
	samples = list()

	# number of pos(neg) samples left to be picked
	num_pos = int(F/2)
	num_neg = int(F/2)

	np.random.shuffle(train_data)

	for sample in train_data:
		label = sample[-1]

		if((label==1 and num_pos==0) or (label==0 and num_neg==0)):
			continue

		if(label==1):
			num_pos -= 1
		else:
			num_neg -= 1

		samples.append(sample)

		if(num_pos==0 and num_neg==0):
			break

	samples = np.asarray(samples, dtype=np.float64)
	return samples


# construct a random forest having M trees
def construct_perceptron_forest(train, M=10):
	forest = list()
	# F = int(train.shape[0]/10)
	F = 1000
	for i in range(M):
		subset = random_sample(train,F)
		tree = build_tree(subset)
		forest.append(tree)

	return forest

# the hash code function for a test sample u
def hash_code(forest, u):
	# sum of positive posterior probabilties
	p_sum = 0.0
	# sum of negative posterior probabilties
	n_sum = 0.0
	for tree in forest:
		label, num_pos, num_samps = prediction(u, tree)
		if(label==1):
			p_sum += (num_pos*1.0)/num_samps
		else:
			n_sum += ((num_samps-num_pos)*1.0)/num_samps


	return 1 if p_sum-n_sum>=0.0 else -1


# generate the l-dimensional binary code vector for each test sample
def binary_codes_test(forests, test):
	test_codes = list() # num_test x l
	for u in test:
		binary_code = list() # 1 x l
		for forest in forests:
			binary_code.append(hash_code(forest, u))

		test_codes.append(binary_code)

	return test_codes

# train/update the perceptron forest to generate forests
def binary_codes_train(train, l=100, M=10):
	# construct M forests
	M_forests = list()
	for j in range(l):
		forest = construct_perceptron_forest(train)
		M_forests.append(forest)

	return M_forests


# construct training data for training the perceptron forest
# frame is the current frame, loc is the location of the object
# in the current frame as predicted by the classifier
def construct_training_data_forest(frame, loc, alpha, beta, patch_size=16):
	patches_pos, patches_neg = get_pos_neg_features(frame, loc, alpha, beta, patch_size)
	r,c,   = patches_pos.shape
	temp_pos = np.ones((r,c+1))
	temp_pos[:,:-1] = patches_pos
	train_pos = temp_pos

	print("r,c=",r,c)

	r,c  = patches_neg.shape
	temp_neg = np.zeros((r,c+1))
	temp_neg[:,:-1] = patches_neg
	train_neg = temp_neg

	train = np.concatenate((train_pos, train_neg))
	np.random.shuffle(train)

	return train

def get_pos_neg_features(img, loc, alpha, beta, patch_size):
	# euclidean distance between two points
	e_dist = lambda p1, p2 : np.sqrt(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))

	# coordinates all points of the frame within a given distance range from a point
	def find_points(r,c,y,x,dist1, dist2):
		# return [(i,j) for j in range(c) for i in range(r) if (e_dist((i,j),(x,y)) <= dist2 and (e_dist((i,j),(x,y)) >= dist1))]
		dist_list = list()
		for i in range(r):
			for j in range(c):
				dist = e_dist((i,j),(y,x))
				if(dist >= dist1 and dist < dist2):
					dist_list.append((i,j))
		return dist_list

	x_obj, y_obj = loc
	r, c, ch = img.shape
	d = int(patch_size/2)

	def out_of_bounds(y,x):
		# print("In here:",r,c)
		return (x<0 or y<0 or x>c or y>r)

	# pos patches
	# patch_features stores the feature vectors corr to each patch
	patches_pos = list()
	alpha_dist = find_points(r,c,y_obj,x_obj,0,alpha)

	# print(loc)
	alpha_points = len(alpha_dist)
	print("Alpha points:", alpha_points)

	print("Extracting pos patches")
	# print(img.shape)
	# for point in tqdm(alpha_dist):
	for point in tqdm(alpha_dist):
		y, x = point
		# print(y,x)
		# take a patch centered at this point
		# if any part of the patch falls out of the image, ignore the patch
		if(out_of_bounds(y-d,x-d) or out_of_bounds(y-d,x+d) or out_of_bounds(y+d,x-d) or out_of_bounds(y+d,x+d)):
			# print("Really", (y,x))
			continue
		
		patch = img[y-d:y+d,x-d:x+d,:]
		# print(patch.shape)
		train_point = compute_features(patch)
		if(train_point.size < 800):
			print("Culprits:",(x,y))
		# print(patch.shape)
		# print("train_point", train_point.shape)
		patches_pos.append(train_point)

	patches_pos = np.asarray(patches_pos)
	print("patches_pos", patches_pos.shape)

	# neg patches
	# patch_features stores the feature vectors corr to each patch
	patches_neg = list()
	beta_dist = np.asarray(find_points(r,c,y_obj,x_obj,alpha,beta))
	beta_points = len(beta_dist)
	beta_inds = np.random.choice(np.arange(start=0, stop=beta_points), size=alpha_points, replace=False)
	beta_dist = list(beta_dist[beta_inds])

	print("Extracting neg patches")
	for point in tqdm(beta_dist):
		y, x = point
		# take a patch centered at this point
		# if any part of the patch falls out of the image, ignore the patch
		if(out_of_bounds(y-d,x-d) or out_of_bounds(y-d,x+d) or out_of_bounds(y+d,x-d) or out_of_bounds(y+d,x+d)):
			continue

		patch = img[y-d:y+d,x-d:x+d, :]
		train_point = compute_features(patch)
		patches_neg.append(train_point)

	patches_neg = np.asarray(patches_neg)

	return patches_pos, patches_neg


# compute the confidence scores of the samples in the new frame using the classifier 
# trained on the samples of the previous frame
def compute_confidence_scores(classifier, forests, frame, loc, get_codes=False):
	coords, patch_features = get_patches(frame, loc)
	patch_codes = binary_codes_test(forests, patch_features)
	# find the index of the patch getting maximum score from the classifier
	scores = classifier.decision_function(patch_codes)
	ind = np.argmax(scores)
	# return the new (predicted) location of the target
	# if(get_codes):
		# return coords[ind] , (patch_codes, scores)
	# else
		# return coords[ind]
	return (coords, patch_codes, scores)
	


# given training data (from patches around a predicted point), train/update the classifier 
# in an online fashion
def classifier_train(train):
	all_classes = np.array([-1, 1])
	X_train = train[:,:-1]
	y_train = train[:,-1]
	classifier = clfPA.partial_fit(X_train, y_train, classes=all_classes)
	return classifier


# given the current frame image and the predicted image location loc,
# extract the patches around this location, compute their codes using 
# trained the perceptron forests and form the training data for the classifier
# Training data of the classifier: all patches at distance < alpha are pos
# all patches at alpha < distance < beta are neg
def construct_training_data_classifier(forests, frame, loc, alpha, beta, patch_size=16):
	patches_pos, patches_neg = get_pos_neg_features(frame, loc, alpha, beta, patch_size)

	codes_pos = binary_codes_test(forests, patches_pos)
	# append a column of ones to indicate pos label
	temp = np.ones(r,c+1)
	temp[:,:-1] = codes_pos
	train_pos = temp

	codes_neg = binary_codes_test(forests, patches_neg)
	# append a column of -1s to indicate neg label
	temp = np.ones(r,c+1)*(-1)
	temp[:,:-1] = codes_neg
	train_neg = temp

	train = np.concatenate((train_pos, train_neg))
	np.random.shuffle(train)

	return train

# given the initial confidence score vector s0, refines the scores and returns them
def hypergraph_propagation(coords, patch_codes, scores, tau=50):
	# incidence matrix
	H = patch_codes.copy()
	H[H<0] = 0
	assert H.shape == (patch_codes.shape[0],100)
	# Dv
	Dv = np.diag(np.sum(H,axis=1))
	Dvinv = np.linalg.inv(Dv)
	# De
	De = np.diag(np.sum(H,axis=0))
	Deinv = np.linalg.inv(De)

	# TPM
	P = np.matmul(Dvinv,np.matmul(H, np.matmul(Deinv, H.T)))
	# alpha
	alpha = 0.1

	scores = np.squeeze(scores)
	curr_scores = scores
	for i in range(tau):
		curr_scores = alpha*P*curr_scores + (1-alpha)*scores

	ind = np.argmax(curr_scores)
	return coords[ind]


# construct a bounding box around the predicted location and save the image
def construct_bounding_box(img, img_name):
	# x, y, width, height = cv2.selectROI(img)
	x_cv = 155
	y_cv = 64
	width = 63
	height = 69
	# print(x_cv,y_cv,width,height)

	cv2.rectangle(img, (x_cv,y_cv), (x_cv+width,y_cv+height), (0,0,255), 3)
	img_name = os.path.join(output_dir,img_name)
	# print(img_name)
	cv2.imwrite(img_name, img)
	return int(x_cv+width/2), int(y_cv+height/2)

def main():
	img_names_list = list()
	for img_name in os.listdir(frames_dir):
		if('jpg' or 'png' in img_name):
			img_names_list.append(img_name)
	img_names_list.sort()

	patch_size = 16
	alpha = 16
	beta = 32

	img0 = cv2.imread(os.path.join(frames_dir, img_names_list[0]))
	loc_cv = construct_bounding_box(img0, img_names_list[0])
	height, width, channels = img0.shape

	# train the forests and the classifier on the first frame
	print("Frame 0:")
	print("Computing Features")
	# features_data = construct_training_data_forest(img0,loc_cv,alpha,beta)
	# np.save('features0.npy', features_data)
	features_data = np.load('features0.npy')
	print("Training Forests")
	forests = binary_codes_train(features_data)
	print("Computing Codes")
	codes_data = construct_training_data_classifier(forests, img0, loc_cv, alpha, beta, patch_size)
	print("Training Classifier")
	classifier = classifier_train(codes_data)

	for i, img_name in tqdm(enumerate(img_names_list)):
		if(i==0):
			continue
		print("Frame %d" %i)
		frame = cv2.imread(os.path.join(frames_dir,img_name), cv2.IMREAD_COLOR)

		print("Predicting")
		coords, patches_codes, scores = compute_confidence_scores(classifier, forests, frame, loc)
		loc = hypergraph_propagation(coords, patches_codes, scores)
		print("Saving Image")
		construct_bounding_box(frame, loc, patch_size, img_name)

		if(i%forest_update==0):
			print("Updating Perceptron Forests")
			features_data = construct_training_data_forest(frame,loc,alpha,beta)
			forests = binary_codes_train(features_data)

		if(i%classifier_update==0):
			print("Updating Classifier")
			codes_data = construct_training_data_classifier(forests, frame, loc, alpha, beta, patch_size)
			classifier = classifier_train(codes_data)


if __name__ == "__main__":
	main()