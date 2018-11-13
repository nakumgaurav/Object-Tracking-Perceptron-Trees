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
