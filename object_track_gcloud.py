import os
import numpy as np
import cv2
import skimage
import argparse
from skimage.feature import local_binary_pattern, hog, haar_like_feature
from skimage.transform import integral_image
from pdt import build_tree, prediction
from sklearn.linear_model import PassiveAggressiveClassifier as clfPA
from sklearn.linear_model import LogisticRegression as clfLR
from sklearn.linear_model import SGDClassifier as clfSGD
from sklearn.linear_model import Perceptron as clfP
from sklearn.svm import SVC
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-f', "--frames_dir", default='new_frames/bag')
# parser.add_argument('-g', "--ground_truth_dir", default='/home/gaurav/Desktop/test/gt/bag')
parser.add_argument('-o', "--output_dir", default='output')
args = parser.parse_args()

frames_dir = args.frames_dir
# ground_truth_dir = args.ground_truth_dir
output_dir = args.output_dir

forest_update = 10
classifier_update = 5

EPS = 1e-12
classifier1 = clfPA(max_iter=1000, random_state=10, loss='squared_hinge', tol=1e-4)
classifier2 = clfLR(solver='sag', tol=1e-1)
classifier3 = clfP(tol=1e-3)
classifier4 = clfSGD(average=True, max_iter=100)
classifier5 = SVC(gamma='auto')

params = {'initRad': 3, 'initMaxNegNum': 65, 'searchWinsize': 25, 'trackInPosRad': 4, 
'trackMaxNegNum': 65, 'trackMaxPosNum': 10000, 'detectWinSize': 20}

MODE_INIT_POS = 0
MODE_INIT_NEG = 1
MODE_TRACK_POS = 2
MODE_TRACK_NEG = 3
MODE_DETECT = 4

def sampling_impl(img, bounding_box, mode):
    inrad = 0.0
    outrad = 0.0
    maxnum = 0

    samples = None
    x, y, w, h = bounding_box
    if(mode==MODE_INIT_POS):
        inrad = params['initRad']
        samples = sample_image(img, x, y, w, h, inrad)
    elif(mode==MODE_INIT_NEG):
        inrad = 2.0 * params['searchWinsize']
        outrad = 1.5 * params['initRad']
        maxnum = params['initMaxNegNum']
        samples = sample_image(img, x, y, w, h, inrad, outrad, maxnum)
    elif(mode==MODE_TRACK_POS):
        inrad = params['trackInPosRad']
        outrad = 0.0
        maxnum = params['trackMaxPosNum']
        samples = sample_image(img, x, y, w, h, inrad, outrad, maxnum)
    elif(mode==MODE_TRACK_NEG):
        inrad = 1.5 * params['searchWinsize']
        outrad = params['trackInPosRad'] + 5
        maxnum = params['trackMaxNegNum']
        samples = sample_image(img, x, y, w, h, inrad, outrad, maxnum)
    elif(mode==MODE_DETECT):
        inrad = params['detectWinSize']
        samples = sample_image(img, x, y, w, h, inrad)
    else:
        inrad = parms['initRad']
        samples = sample_image(img, x, y, w, h, inrad)


    return samples

def sample_image(img, x_bb, y_bb, w, h, inrad, outrad=0, maxnum=10000):
    # euclidean distance between two points
    e_dist = lambda p1, p2 : np.sqrt(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))

    coords = list()
    r, c = img.shape

    rowsz = r - h - 1
    colsz = c - w - 1

    inrad_sq = inrad*inrad
    outrad_sq = outrad*outrad

    minrow = int(max(0, y_bb - inrad))
    maxrow = int(min(rowsz-1, y_bb + inrad))
    mincol = int(max(0, x_bb - inrad))
    maxcol = int(min(colsz-1, x_bb + inrad))

    sample_size = (maxrow - minrow + 1) * (maxcol - mincol + 1)
    # print("sample_size=", sample_size)

    prob = float(maxnum)/sample_size
    print("prob=", prob)
    samples = list()

    for y in range(minrow, maxrow+1):
        for x in range(mincol, maxcol+1):
            dist = e_dist((y_bb,x_bb),(y,x))
            if(np.random.uniform() < prob and dist<inrad_sq and dist>=outrad_sq):
            # if(dist<inrad_sq and dist>=outrad_sq):
                patch = img[y:y+h,x:x+w]
                train_point = np.resize(patch,(patch.size,))
                samples.append(train_point)
                coords.append((x,y))

    # print(samples[0])
    print("samples.size=", len(samples))
    assert len(samples) < maxnum
    samples = np.asarray(samples, np.float)

    return samples, coords

# generate a random sample having equal number of positive and negative samples
# total F samples are generated
# by convention last column of the data matrix consists of the labels
def random_sample(train, F=100):
	train_data = np.copy(train)

	num_pos = np.sum(train_data[:,-1]==1)
	num_neg = np.sum(train_data[:,-1]==0)

	# assert num_neg>=num_pos

	num_pos = min(int(F/2), num_pos, num_neg)
	num_neg = num_pos

	samples = list()

	# number of pos(neg) samples left to be picked
	# num_pos = int(F/2)
	# num_neg = int(F/2)

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
		print("Tree # ",i+1)
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
		# print (label, num_pos, num_samps)
		if(label==1):
			p_sum += (num_pos*1.0)/num_samps
		else:
			n_sum += ((num_samps-num_pos)*1.0)/num_samps


	# return 1 if p_sum-n_sum>=0.0 else -1
	return p_sum-n_sum


# generate the l-dimensional binary code vector for each test sample
def binary_codes_test(forests, test):
	def normalize_codes(vec):
		# mini = np.min(vec)
		# vec = vec + np.abs(mini)
		vec = vec/(np.max(np.abs(vec)) + EPS)
		return vec

	test_codes = list() # num_test x l
	# print(test.shape)
	for u in tqdm(test):
		binary_code = list() # 1 x l
		for forest in forests:
			binary_code.append(hash_code(forest, u))

		test_codes.append(normalize_codes(binary_code))

	test_codes = np.array(test_codes)
	# print(test_codes)
	return test_codes

# train/update the perceptron forest to generate forests
def binary_codes_train(train, l=100, M=10):
	# construct M forests
	M_forests = list()
	for j in range(l):
		print("\nForest #", j+1)
		forest = construct_perceptron_forest(train)
		M_forests.append(forest)

	return M_forests


def append_labels(data, label):
    # print(data.shape)
    r,c  = data.shape
    print("r,c=",r,c)
    temp = np.zeros((r,c+1))
    temp[:,:-1] = data
    temp[:,-1] = label
    return temp

def match_dims(p1, p2):
    sh1 = p1.shape
    sh2 = p2.shape
    print("sh1=", sh1)
    print("sh2=", sh2)
    mini = sh1[1]
    maxi = sh2[1]
    if(mini==maxi):
        return
    else:
        stretch =  None
        diff = np.abs(maxi-mini)
        if(mini<maxi):
            stretch = p1[:,-1]
        else:
            stretch = p2[:,-1]

        stretch = np.reshape(stretch, (stretch.size, 1))
        stretch = np.repeat(stretch, diff, axis=1)

        if(mini<maxi):
            temp = np.zeros(sh1[0], sh1[1] + diff)
            temp[:,:sh1[1]] = p1
            temp[:, sh1[1]:] = stretch
            p1 = temp
        else:
            temp = np.zeros(sh2[0], sh2[1] + diff)
            temp[:,:sh1[1]] = p2
            temp[:, sh1[1]:] = stretch
            p2 = temp

# construct training data for training the perceptron forest
# frame is the current frame, loc is the location of the object
# in the current frame as predicted by the classifier
def construct_training_data_forest(frame, bb, mode):
    patches = patches_pos = patches_neg = None
    if(mode=="INIT"):
        patches_pos, _ = sampling_impl(frame, bb, mode=MODE_INIT_POS)
        patches_neg, _ = sampling_impl(frame, bb, mode=MODE_INIT_NEG)
        # match_dims(patches_pos, patches_neg)


    elif(mode=="TRACK"):
        patches_pos, _ = sampling_impl(frame, bb, mode=MODE_TRACK_POS)
        patches_neg, _ = sampling_impl(frame, bb, mode=MODE_TRACK_NEG)

    else:
        print("Invalid Mode")

    patches_pos = append_labels(patches_pos, 1)
    patches_neg = append_labels(patches_neg, 0)
    patches = np.concatenate((patches_pos, patches_neg))
    np.random.shuffle(patches)
    return patches
 

def construct_training_data_classifier(forests, frame, bb, mode):
    patches_pos = patches_neg = None
    if(mode=="INIT"):
        patches_pos, _ = sampling_impl(frame, bb, mode=MODE_INIT_POS)
        patches_neg, _ = sampling_impl(frame, bb, mode=MODE_INIT_NEG)

    elif(mode=="TRACK"):
        patches_pos, _ = sampling_impl(frame, bb, mode=MODE_TRACK_POS)
        patches_neg, _ = sampling_impl(frame, bb, mode=MODE_TRACK_NEG)

    # elif(mode=="PREDICT"):
        # patches = sampling_impl(frame, bb, mode=MODE_DETECT)
    else:
        print("Invalid Mode")
        return

    codes_pos = append_labels(binary_codes_test(forests, patches_pos), 1)
    codes_neg = append_labels(binary_codes_test(forests, patches_neg), 0)
    codes = np.concatenate((codes_pos, codes_neg))
    np.random.shuffle(codes)
    return codes


# given training data (from patches around a predicted point), train/update the classifier 
# in an online fashion
def classifier_train(train):
	all_classes = np.array([-1, 1])
	X_train = train[:,:-1]
	y_train = train[:,-1]
	# classifier.partial_fit(X_train, y_train, classes=all_classes)
	classifier1.fit(X_train, y_train)
	classifier2.fit(X_train, y_train)
	classifier3.fit(X_train, y_train)
	classifier4.fit(X_train, y_train)
	classifier5.fit(X_train, y_train)
	classifier = (classifier1, classifier2, classifier3, classifier4, classifier5)
	return classifier


# compute the confidence scores of the samples in the new frame using the classifier
# trained on the samples of the previous frame
def compute_confidence_scores(classifier, forests, frame, bb):
    patches, coords = sampling_impl(frame, bb, mode=MODE_DETECT)
    patch_codes = binary_codes_test(forests, patches)
    # find the index of the patch getting maximum score from the classifier
    classifier1, classifier2, classifier3, classifier4, classifier5 = classifier


    def normalize(scores):
        return scores/np.max(np.abs(scores))

    scores1 = normalize(classifier1.decision_function(patch_codes))
    scores2 = normalize(classifier2.decision_function(patch_codes))
    scores3 = normalize(classifier3.decision_function(patch_codes))
    scores4 = normalize(classifier4.decision_function(patch_codes))
    scores5 = normalize(classifier5.decision_function(patch_codes))

    scores = (scores1+scores2+scores3+scores4+scores5)*1.0/5.0
    # ind = np.argmax(scores)
    # return the new (predicted) location of the target
    # return coords[ind] #, (coords,scores)
    return (coords, patch_codes, scores)

# given the initial confidence score vector s0, refines the scores and returns them
def hypergraph_propagation(coords, patch_codes, scores, tau=50):
	scores = np.asarray(scores, np.float)
	# incidence matrix
	H = patch_codes.copy()
	H[H<0] = 0.0
	H = np.asarray(H, np.float)
	assert H.shape == (patch_codes.shape[0],100)
	# Dv
	v = np.sum(H, axis=1)
	v = np.asarray(v, dtype=np.float)
	v[v==0] = 1.0
	# for a in v:
		# print(a)
	# print(v==0)
	Dv = np.diag(v)
	Dvinv = np.linalg.inv(Dv)
	# De
	e = np.sum(H,axis=0)
	e = np.asarray(e, dtype=np.float)
	e[e==0] = 1.0
	De = np.diag(e)
	Deinv = np.linalg.inv(De)

	# TPM
	P = np.matmul(Dvinv,np.matmul(H, np.matmul(Deinv, H.T)))
	# alpha
	alpha = 0.99

	scores = np.squeeze(scores)
	curr_scores = scores.copy()
	for i in range(tau):
		curr_scores = alpha*np.matmul(P,curr_scores) + (1-alpha)*scores

	print(curr_scores.shape)
	ind = np.argmax(curr_scores)
	return coords[ind]


# draw a bounding box around the object for the first frame and save the image
def construct_bounding_box(img, img_name):
    # x, y, width, height = cv2.selectROI(img)
    # 155 65 62 65
    x = 79
    y = 34
    width = 28
    height = 30
    # print(x_cv,y_cv,width,height)

    cv2.rectangle(img, (x,y), (x+width,y+height), (0,0,255), 1)
    img_name = os.path.join(output_dir,img_name)
    # print(img_name)
    img = img*255.0
    img = np.asarray(img, dtype=np.uint8)
    cv2.imwrite(img_name, img)
    # print("Min:", np.min(img),  "Max:", np.max(img))
    print("should be ", x, y, width, height)
    return x, y, width, height

# loc is the location of the top-left corner of the bounding box
def	save_img(img, loc, siz, img_name):
    x, y = loc
    w, h = siz
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
    img_name = os.path.join(output_dir,img_name)
    img = img*255.0
    img = np.asarray(img, dtype=np.uint8)    
    cv2.imwrite(img_name, img)



def main():
    img_names_list = list()
    for img_name in os.listdir(frames_dir):
        if('jpg' or 'png' in img_name):
            img_names_list.append(img_name)
    img_names_list.sort()

    img0 = cv2.imread(os.path.join(frames_dir, img_names_list[0]))
    img0 = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0 = cv2.normalize(img0, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bb = construct_bounding_box(img0, img_names_list[0])
    x, y, w, h = bb
    loc = (x, y)

    # train the forests and the classifier on the first frame
    print("Frame 0:")
    print("Computing Patches/Features")
    # patches_data = construct_training_data_forest(img0,bb,mode="INIT")
    # print("patches_data:", patches_data.shape)
    # np.save('patches0.npy', patches_data)
    patches_data = np.load('patches0.npy')

    # print("Training Forests")
    # forests = binary_codes_train(patches_data)

    # Save the forest for frame 1
    # pickle_out = open("forests0.pickle", "wb")
    # pickle.dump(forests, pickle_out, protocol=2)
    # pickle_out.close()

    pickle_in = open("forests0.pickle","rb")
    forests = pickle.load(pickle_in)

    # print("Computing Codes")
    # codes_data = construct_training_data_classifier(forests, img0, bb, "INIT")

    # Save the codes for frame 1
    # pickle_out = open("codes0.pickle", "wb")
    # pickle.dump(codes_data, pickle_out, protocol=2)
    # pickle_out.close()

    pickle_in = open("codes0.pickle","rb")
    codes_data = pickle.load(pickle_in)

    print("Initial Location:", loc)
    print("Training Classifier")
    classifier = classifier_train(codes_data)

    for i, img_name in tqdm(enumerate(img_names_list)):
        if(i==0):
            continue
        print("Frame %d" %i)
        frame = cv2.imread(os.path.join(frames_dir,img_name), cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.normalize(frame, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        print("Predicting")
        # loc_cv = compute_confidence_scores(classifier, forests, frame, loc_cv, patch_size)
        coords, patches_codes, scores = compute_confidence_scores(classifier, forests, frame, bb)
        loc_cv = coords[np.argmax(scores)]
        # print("Without HP:", x1, y1)
        # for score in scores:
        # 	print(score)
        # loc_cv = hypergraph_propagation(coords, patches_codes, scores)
        print("New Location:", loc_cv)
        bb = (loc_cv[0], loc_cv[1], w, h)

        print("Saving Image")
        save_img(frame, loc_cv, (w,h), img_name)

        if(i%forest_update==0):
            print("Updating Perceptron Forests")
            patches_data = construct_training_data_forest(frame,bb, mode='TRACK')
            forests = binary_codes_train(patches_data)

        if(i%classifier_update==0):
            print("Updating Classifier")
            codes_data = construct_training_data_classifier(forests, frame, bb, "TRACK")
            classifier = classifier_train(codes_data)


if __name__ == "__main__":
	main()




# Try with 1. unnormalized hash codes 2. changing bounding box size