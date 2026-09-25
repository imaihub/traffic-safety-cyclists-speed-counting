# Licence of this sample dataset

The contents of this folder are derived from the **Stanford Drone Dataset** (Stanford Computational Vision and
Geometry Lab, https://cvgl.stanford.edu/projects/uav_data/) and are licensed under the
**Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (CC BY-NC-SA 3.0)**:
https://creativecommons.org/licenses/by-nc-sa/3.0/

This licence applies to everything in this folder (images, video, annotation and ground-truth files, scene and
settings files). It is separate from the licence of the Cyclist Tracker software (GNU AGPL v3, see the LICENSE file
in the root of the repository).

## Attribution

Original work: *Stanford Drone Dataset*, A. Robicquet, A. Sadeghian, A. Alahi and S. Savarese, Stanford University.
Cite: A. Robicquet, A. Sadeghian, A. Alahi, S. Savarese, "Learning Social Etiquette: Human Trajectory Prediction in
Crowded Scenes", ECCV 2016.

The metric scale used for the reference speeds comes from the OpenTraj project
(https://github.com/crowdbotp/OpenTraj, `datasets/SDD/estimated_scales.yaml`).

## Changes made to the original

* Selected scene `gates`, videos 4 (test), 5 (validation) and 7 (training).
* Kept every 6th frame (test) or every 30th frame (training, validation); cropped every frame to the region
  x 200–1300, y 702–1982 (original pixels) and resized it by a factor 0.5; re-encoded as JPEG and as an H.264 video.
* Kept only annotations of class `Biker` that are in view; converted them to COCO and MOT-style text formats with
  coordinates in the cropped and resized frames.
* Added derived files: reference speeds and crossings (computed from the annotations and the metric scale), a
  camera scene and analysis settings.

## Terms in short

You may share and adapt this material for **non-commercial** purposes, provided you give appropriate credit,
indicate changes, and distribute your contributions under the same licence. See the licence text for the full terms.
