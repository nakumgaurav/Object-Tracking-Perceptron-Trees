Use:
python "xyz".py --frames_dir frames/ --output_dir output/

frames_dir should contain the frames of a particular sequence
output_dir would store the frames with predicted bounding boxes marked in green (except the first frame)

Replace xyz with any of the files starting with "object_track":

1. object_track.py -> using local patches, raw pixel intensities
2. object_track_boxes.py -> using full target sized bounding boxes
3. object_track_features.py -> using local patches, 4 types of features
4. object_track_gcloud.py -> gcloud version of (3)
5. object_track_invariant.py -> rotation and scale invariant version of (2)

For independently testing perceptron tree: 
python pdt.py num_epochs
(must have a dataset)

Eg:
python pdt.py 50

HyperParameter values used for experiment:
In a way similar to the MIL tracker, the search radius for cropping positive patches for the perceptron forest and the classifier is set to $\alpha = 16$ pixels, and search radius for negative patches is set to beta = 48 pixels. We use raw pixel intensities as our features to be fed to the perceptron forest due to their ease of implementation and computational effectiveness. For perceptron forest hashing, the binary code length 'l' is chosen as 100. The training sample sizes N and F are chosen as 100 and 50, respectively. Each perceptron forest has 10 trees with maximum depth 2. The perceptron forests are updated after every 10 frames, while the PA classifier is updated after every 5 frames. 

In the optical flow based proposal generation algorithm, pixels that are atleast 50 pixels apart are chosen for a reliable estimation. Also, we limit the number of candidates generated to 1000. The PSO algorithm runs for 5 iterations with a population size (number of particles per iteration) of 25.
