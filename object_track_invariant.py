import os
import math
import collections
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
import PSO

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

Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

fixed_shape = None

def sampling_impl(img, bounding_box, mode):
    inrad = 0.0
    outrad = 0.0
    maxnum = 0

    samples = None
    x, y, w, h, theta = bounding_box
    if(mode==MODE_INIT_POS):
        inrad = params['initRad']
        samples = sample_image(img, x, y, w, h, theta, inrad)
    elif(mode==MODE_INIT_NEG):
        inrad = 2.0 * params['searchWinsize']
        outrad = 1.5 * params['initRad']
        maxnum = params['initMaxNegNum']
        samples = sample_image(img, x, y, w, h, theta, inrad, outrad, maxnum)
    elif(mode==MODE_TRACK_POS):
        inrad = params['trackInPosRad']
        outrad = 0.0
        maxnum = params['trackMaxPosNum']
        samples = sample_image(img, x, y, w, h, theta, inrad, outrad, maxnum)
    elif(mode==MODE_TRACK_NEG):
        inrad = 1.5 * params['searchWinsize']
        outrad = params['trackInPosRad'] + 5
        maxnum = params['trackMaxNegNum']
        samples = sample_image(img, x, y, w, h, theta, inrad, outrad, maxnum)
    elif(mode==MODE_DETECT):
        inrad = params['detectWinSize']
        samples = sample_image(img, x, y, w, h, theta, inrad)
    else:
        inrad = parms['initRad']
        samples = sample_image(img, x, y, w, h, theta, inrad)

    return samples

def get_patch(image,width,height, theta, centroid):
    cx, cy = centroid
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = int(cx - v_x[0] * (width / 2) - v_y[0] * (height / 2))
    s_y = int(cy - v_x[1] * (width / 2) - v_y[1] * (height / 2))

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(int(width), int(height)),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

    
def rotate(self,origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = int(ox + math.cos(angle) * (px - ox) - math.sin(angle) * (oy - py))
    qy = int(oy + math.sin(angle) * (px - ox) + math.cos(angle) * (oy - py))
    return qx, qy 

def compute_points(x,y,w,h,theta):

    xmax=abs(x+round(w/2))
    xmin=abs(x-round(w/2))

    ymax=abs(y+round(h/2))
    ymin=abs(y-round(h/2))      
    
    pt1 = rotate((x,y),(xmin,ymin),theta)
    pt2 = rotate((x,y),(xmax,ymin),theta)
    pt3 = rotate((x,y),(xmax,ymax),theta)
    pt4 = rotate((x,y),(xmin,ymax),theta)

    return (pt1, pt2, pt3, pt4)

def compute_centroid(points):
    cx, cy = (0, 0)
    for point in points:
        x, y = point
        cx += x
        cy += y
    cx /= 4
    cy /= 4
    return (int(cx), int(cy))

def sample_image(img, xc, yc, w, h, theta, inrad, outrad=0, maxnum=10000):
    # euclidean distance between two points
    e_dist = lambda p1, p2 : np.sqrt(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))

    # print("x_bb, y_bb, w, h, theta, inrad", x_bb, y_bb, w, h, theta, inrad)
    global fixed_shape
    shp = fixed_shape
    coords = list()
    r, c = img.shape

    # print("r, c", r, c)

    rowsz = r - h - 1
    colsz = c - w - 1

    # print("rowsz, colsz", rowsz, colsz)

    inrad_sq = inrad*inrad
    outrad_sq = outrad*outrad

    # print("inrad_sq, outrad_sq", inrad_sq, outrad_sq)

    minrow = int(max(0, yc - inrad))
    maxrow = int(min(rowsz-1, yc + inrad))
    mincol = int(max(0, xc - inrad))
    maxcol = int(min(colsz-1, xc + inrad))

    # print("minrow, maxrow, mincol, maxcol, curr_cx, curr_cy", minrow, maxrow, mincol, maxcol, curr_cx, curr_cy)

    def out_of_bounds(points):
        # print("In here:",r,c)
        for point in points:
            x, y = point
            if(x<0 or y<0 or x>c or y>r):
                return True
        return False

    sample_size = (maxrow - minrow + 1) * (maxcol - mincol + 1)
    # print("sample_size=", sample_size)

    prob = float(maxnum)/sample_size
    print("prob=", prob)
    samples = list()

    for y in range(minrow, maxrow+1):
        for x in range(mincol, maxcol+1):
            dist = e_dist((y_bb,x_bb),(y,x))
            points = compute_points(x,y,w,h,theta)

            if(np.random.uniform() < prob and dist<inrad_sq and dist>=outrad_sq and not out_of_bounds(points)):
                patch = get_patch(img, w, h, theta, (x,y))
                if(w != shp[1] or h != shp[0]):
                    patch = cv2.resize(patch, None, fx=shp[1]/w, fy=shp[0]/h, interpolation=cv2.INTER_LANCZOS4)
                train_point = np.resize(patch,(patch.size,))
                samples.append(train_point)
                coords.append((x,y,w,h,theta))

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

	assert num_neg>=num_pos

	num_pos = min(int(F/2), num_pos)
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
	for u in test:
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
    all_classes = np.array([0, 1])
    X_train = train[:,:-1]
    y_train = np.asarray(train[:,-1], dtype=np.int)
    classifier1.partial_fit(X_train, y_train, classes=all_classes)
    classifier = [classifier1]
    # classifier1.fit(X_train, y_train)
    # classifier2.fit(X_train, y_train)
    # classifier3.fit(X_train, y_train)
    # classifier4.fit(X_train, y_train)
    # classifier5.fit(X_train, y_train)
    # classifier = (classifier1, classifier2, classifier3, classifier4, classifier5)
    return classifier



# compute the fitness score of a single patch of the current frame using the classifier and the forests
def compute_fitness_score(classifier, forests, frame, patch_data, shp):
    cx, cy, w, h, theta = patch_data
    patch = get_patch(frame, w, h, theta, (cx,cy))


    patch_gray = patch.copy()
    patch_gray = cv2.cvtColor(patch_gray, cv2.COLOR_BGR2GRAY)

    old_shp = patch.shape
    # print(shp, old_shp)
    # fx = shp[1]/old_shp[1]
    # fy = shp[0]/old_shp[0]
    # print("fx, fy", fx, fy)
    # patch_gray = cv2.resize(patch_gray, None, fx, fy, interpolation=cv2.INTER_LANCZOS4)
    patch_gray = cv2.resize(patch_gray, shp, interpolation=cv2.INTER_LANCZOS4)
    patch_gray = cv2.normalize(patch_gray, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    patch_gray = np.resize(patch_gray, (patch_gray.size,))

    patches = [patch_gray]
    patch_code = binary_codes_test(forests, patches)
    patch_code = np.reshape(patch_code, (1, np.size(patch_code)))

    scores = 0.0
    for clf in classifier:
        temp = clf.decision_function(patch_code)
        scores += temp[0]

    scores = scores/len(classifier)
    return scores

# compute the confidence scores of the samples in the new frame using the classifier
# trained on the samples of the previous frame
def compute_confidence_scores(classifier, forests, frame, bb):
    patches, coords = sampling_impl(frame, bb, mode=MODE_DETECT)
    patch_codes = binary_codes_test(forests, patches)
    # find the index of the patch getting maximum score from the classifier

    def normalize(scores):
        return scores/np.max(np.abs(scores))


    scores = np.zeros((patch_codes.shape[0],))
    for clf in classifier:
        scores += normalize(clf.decision_function(patch_codes))

    scores = scores/len(classifier)

    # ind = np.argmax(scores)
    # return the new (predicted) location of the target
    # return coords[ind] #, (coords,scores)
    return (coords, patch_codes, scores)

# draw a bounding box around the object for the first frame and save the image
def construct_bounding_box(img, img_name):
    # x, y, width, height = cv2.selectROI(img)
    # 155 65 62 65
    x = 155
    y = 65
    width = 62
    height = 65
    # print(x_cv,y_cv,width,height)

    cv2.rectangle(img, (x,y), (x+width,y+height), (0,0,255), 3)
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
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
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
    # img0 = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
    img0_bgr = img0.copy()
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0 = cv2.normalize(img0, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    bb = construct_bounding_box(img0, img_names_list[0])
    x, y, w, h = bb
    # phi = np.tan(float(h)/float(w))
    theta = 0
    bb = (int(x + w/2), int(y + h/2), w, h, theta)
    loc = (x, y)
    global fixed_shape
    fixed_shape = (w, h)

    def init_PSO_Tracker():
        refPt = [Point(x,y), Point(x+w,y), Point(x+w,y+h), Point(x,y+h)]
        selection = Polygon(refPt)
        tracker = PSO.PKTracker(img0_bgr, selection)
        t1=0
        t2=0
        for p in selection.points:
            t1+=p.x
            t2+=p.y
        t1=int(t1*0.25)
        t2=int(t2*0.25)
        tracker.trans_x=np.array([[t1-30,t1+30]])
        tracker.trans_y=np.array([[t2-30,t2+30]])
        return tracker

    tracker = init_PSO_Tracker()

    # train the forests and the classifier on the first frame
    print("Frame 0:")
    print("Computing Patches/Features")
    patches_data = construct_training_data_forest(img0,bb,mode="INIT")
    print("patches_data:", patches_data.shape)
    np.save('patches0.npy', patches_data)
    # patches_data = np.load('patches0.npy')
    print("Training Forests")
    forests = binary_codes_train(patches_data)

    # Save the forest for frame 1
    pickle_out = open("forests0.pickle", "wb")
    pickle.dump(forests, pickle_out, protocol=2)
    pickle_out.close()

    # pickle_in = open("forests0.pickle","rb")
    # forests = pickle.load(pickle_in)

    print("Computing Codes")
    codes_data = construct_training_data_classifier(forests, img0, bb, "INIT")

    # Save the codes for frame 1
    pickle_out = open("codes0.pickle", "wb")
    pickle.dump(codes_data, pickle_out, protocol=2)
    pickle_out.close()

    pickle_in = open("codes0.pickle","rb")
    codes_data = pickle.load(pickle_in)

    print("Initial Location:", loc)
    print("Training Classifier")
    classifier = classifier_train(codes_data)

    for i, img_name in enumerate(img_names_list):
        if(i==0):
            continue
        print("Frame %d" %i)
        frame = cv2.imread(os.path.join(frames_dir,img_name), cv2.IMREAD_COLOR)
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
        frame_bgr = frame.copy()

        print("Predicting")
        loc_cv, w, h, theta = apply_pso(frame_bgr, tracker, classifier, forests, i, img_name)



        # loc_cv = compute_confidence_scores(classifier, forests, frame, loc_cv, patch_size)
        # coords, patches_codes, scores = compute_confidence_scores(classifier, forests, frame, bb)
        # loc_cv = coords[np.argmax(scores)]

        print("New Location:", loc_cv)
        bb = (loc_cv[0], loc_cv[1], w, h, theta)

        if(i%forest_update ==0 or i%classifier_update==0):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.normalize(frame, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if(i%forest_update==0):
            print("Updating Perceptron Forests")
            patches_data = construct_training_data_forest(frame,bb, mode='TRACK')
            forests = binary_codes_train(patches_data)

        if(i%classifier_update==0):
            print("Updating Classifier")
            codes_data = construct_training_data_classifier(forests, frame, bb, "TRACK")
            classifier = classifier_train(codes_data)


def draw_save_img(img, rect, img_name):
    print("Saving Image")
    pt1, pt2, pt3, pt4 = rect.points[0], rect.points[1], rect.points[2], rect.points[3]
    cv2.line(img, pt1, pt2, [0,255,0], 1)
    cv2.line(img, pt2, pt3, [0,255,0], 1)
    cv2.line(img, pt3, pt4, [0,255,0], 1)
    cv2.line(img, pt4, pt1, [0,255,0], 1)
    cv2.imwrite(os.path.join(output_dir, img_name), img)


def apply_pso(frame, tracker, classifier, forests, itr, img_name):
    global fixed_shape
    reg, reg_info = tracker.PSO_shift(frame,itr, forests, classifier, fixed_shape)
    draw_save_img(frame, reg, img_name)
    return reg_info


if __name__ == "__main__":
    main()