params = {'initRad': 3, 'initMaxNegNum': 65, 'searchWinsize': 25, 'trackInPosRad': 4, 
'trackMaxNegNum': 65, 'trackMaxPosNum': 10000}

MODE_INIT_POS = 0
MODE_INIT_NEG = 1
MODE_TRACK_POS = 2
MODE_TRACK_NEG = 3
MODE_DETECT = 4

def sampling_impl(img, bounding_box):
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
		inrad = parms['searchWinsize']
		samples = sample_image(img, x, y, w, h, inrad)
	else:
		inrad = parms['initRad']
		samples = sample_image(img, x, y, w, h, inrad)

def sample_image(img, x_bb, y_bb, w, h, inrad, outrad=0, maxnum=10000):
	r, c = img.shape

	rowsz = r - h - 1
	colsz = c - w - 1

	inrad_sq = inrad*inrad
	outrad_sq = outrad*outrad

	minrow = max(0, y_bb - inrad)
	maxrow = min(rowsz-1, y_bb + inrad)
	mincol = max(0, x_bb - inrad)
	maxcol = min(colsz-1, x_bb + inrad)

	sample_size = (maxrow - minrow + 1) * (maxcol - mincol + 1)

	prob = float(maxnum)/sample_size
	samples = list()

	for y in range(minrow, maxrow+1):
		for x in range(mincol, maxcol+1):
			dist = e_dist((y_bb,x_bb),(y,x))
			if(np.random.uniform() < prob and dist<inrad_sq and dist>=outrad_sq):
				samples.append(img[y:y+h,x:x+h])

	samples = np.asarray(samples, np.float)
	assert samples.size < maxnum

	return samples