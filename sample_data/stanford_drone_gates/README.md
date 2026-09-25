# Sample dataset: Stanford Drone Dataset, scene "gates"

A small, **real** dataset in exactly the structure that Cyclist Tracker expects (see `docs/DATASET.md`). Use it
as a reference when you prepare your own data, and to try every function of the program with real footage.

It is an excerpt of the [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) (SDD): a drone
hovering above the Stanford University campus, filming a road with a roundabout that is used by many cyclists.
All boxes and track IDs are the original **human annotations** of the SDD authors, converted to our formats.

> **Licence: CC BY-NC-SA 3.0 — non-commercial use only.** This folder is *not* covered by the AGPL licence of the
> software. See [LICENSE.md](LICENSE.md). Delete the folder if you need a repository without it.

## Contents

```
stanford_drone_gates/
├── train/          74 frames of gates/video7, one frame per second
├── val/            70 frames of gates/video5, one frame per second
├── test/           367 frames of gates/video4, every 6th frame (29.97 fps -> 5 fps), 73 seconds
├── annotations/
│   ├── train.json  250 cyclist boxes   (COCO, with track_id attribute)
│   ├── val.json    276 cyclist boxes
│   └── test.json   2348 cyclist boxes, 53 different cyclists
├── ground_truth/
│   ├── tracking_test.txt   all test boxes with track IDs
│   ├── speeds_test.csv     speeds of the 20 cyclists that ride between line A and line B
│   └── counts_test.csv     the same 20 crossings (12 to the roundabout, 8 from the roundabout)
├── scene.json      region of interest, counting lines and calibration
├── settings.json   analysis settings for this camera (5 fps, box centre as follow point)
├── test_frames.txt list of all test frames
└── gates_video4_5fps.mp4   the test frames as a video (to try video input)
```

Note how the dataset follows the rules of `docs/DATASET.md`: **train, val and test come from different videos**
(no near-identical frames in two splits), training frames may be sparse (1 fps), while the test frames are
consecutive at a constant frame rate. Only bikers are annotated; pedestrians, skateboarders and carts are not.

## How the ground truth was made

* **Boxes and IDs** – the SDD annotations of class `Biker` that are in view (`lost = 0`). SDD creates part of
  the boxes by interpolation between key frames (`generated` attribute in the COCO files), so boxes are not always
  tight.
* **Speeds** – for every biker that passes both counting lines, the exact moments of passing are interpolated from
  the *full-rate* (29.97 fps) annotations. The distance travelled comes from the metres-per-pixel scale of this video
  (0.0441 m per original pixel, from the [OpenTraj](https://github.com/crowdbotp/OpenTraj) `estimated_scales.yaml`).
  These reference speeds are therefore independent of the counting algorithm. The scale is an estimate: absolute
  speeds may be off by a few percent.
* **Frame numbers** in the CSV files count the frames of `test/` from 0, in natural name order.

## Camera and scene

The drone looks straight down, so pixels and metres are related by a single scale and the calibration in
`scene.json` is a rectangle over the whole image. Because the view is top-down, the **centre** of a box is the
position of the cyclist on the ground — that is why `settings.json` sets the follow point to `center` (for a tilted
traffic camera the default `bottom` is correct). Lines A and B cross the lower road 13.2 m apart.

## Try it

GUI: on the Start page press **Open sample dataset**. Command line:

```bash
uv run cyclist-tracker analyze sample_data/stanford_drone_gates/test \
    --scene sample_data/stanford_drone_gates/scene.json --settings sample_data/stanford_drone_gates/settings.json \
    --detector annotations --annotations sample_data/stanford_drone_gates/annotations/test.json --output-dir results
uv run cyclist-tracker evaluate speed --gt sample_data/stanford_drone_gates/ground_truth/speeds_test.csv \
    --pred results/analysis_speeds.csv
```

With the annotated boxes as detections this gives 19 of 20 cyclists counted correctly, an average speed error of
about 0.8 km/h, and IDF1 87 % / MOTA 89 % for tracking.

The pre-trained COCO detector does **not** find these cyclists: it was trained on photos taken from street level,
not from above. This is a real example of why Step 3 (training on your own camera) exists. The `train/` and `val/`
folders show how training data is organised, but with 74 and 70 frames they are far too small to train a good
detector (a quick test did not converge); a real camera needs thousands of annotated frames.

## Rebuilding the sample

The folder is produced by `docs/tools/make_sdd_sample.py` from the original videos. The official download is a
single 71 GB zip; `docs/tools/download_sdd_videos.py` downloads only the three videos that are needed through the
Academic Torrents copy of the dataset.

## Citation

A. Robicquet, A. Sadeghian, A. Alahi, S. Savarese. *Learning Social Etiquette: Human Trajectory Prediction in
Crowded Scenes.* European Conference on Computer Vision (ECCV), 2016.
