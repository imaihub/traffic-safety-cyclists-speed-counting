# Preparing your own dataset

The traffic videos used in the paper (cameras of the Province of Utrecht) are private and are **not** part of this
repository. This guide explains how to prepare your own data so that every function of Cyclist Tracker works with it.

You do not always need everything. Pick the row that matches your goal:

| Goal | Frames | Annotations (COCO) | Ground truth | Scene file |
|---|---|---|---|---|
| Count cyclists and measure speed with the pre-trained COCO detector | test only | – | – | **yes** |
| Count and measure with a detector trained for your camera | test only | – | – | **yes** |
| Train that detector | train + val | **train + val** | – | – |
| Measure tracking quality (IDF1, MOTA) | test | test (with track IDs) | tracking | – |
| Measure speed accuracy (MAE, RMSE, R²) | test | – | speeds | **yes** |
| Measure counting accuracy | test | – | counts | **yes** |

> **Tip.** Two complete examples come with the program:
> * [`sample_data/stanford_drone_gates`](../sample_data/stanford_drone_gates) – **real** drone footage with
>   hand-made annotations of 53 cyclists and ground truth (Stanford Drone Dataset, non-commercial licence).
>   *Open sample dataset* on the Start page loads it.
> * *Try with demo data* in the GUI (or `uv run cyclist-tracker demo demo_data`) creates a synthetic dataset with
>   known speeds.
>
> Both have exactly the structure described here; look at them when you prepare your own data.

---

## 1. Folder structure

```
my_dataset/
├── train/                       frames used to train the detector
│   ├── cam1_20240315_0800_frame_000000.jpg
│   ├── cam1_20240315_0800_frame_000005.jpg
│   └── …
├── val/                         frames used to check the detector during training
├── test/                        frames to analyse and to evaluate on
├── annotations/
│   ├── train.json               boxes for train/  (COCO format)
│   ├── val.json                 boxes for val/
│   └── test.json                boxes for test/   (optional)
├── ground_truth/                optional, only for evaluation (Step 5)
│   ├── tracking_test.txt        boxes with cyclist IDs for test/
│   ├── speeds_test.csv          reference speeds for test/
│   └── counts_test.csv          reference crossings for test/
└── scene.json                   camera configuration (made in Step 2 of the GUI)
```

Create the empty structure with `uv run cyclist-tracker dataset init my_dataset` or the button
*Create empty folder structure* in tab **1 · Dataset**. Check it at any time with
`uv run cyclist-tracker dataset validate my_dataset` or *Check my dataset*: every problem is explained in plain
language, and the check tells you whether the dataset is ready for training and for analysis.

The annotation files of the original experiments were called `training_set_f.json`, `validation_set_f.json` and
`testing_set_f.json`; these names are recognised too.

## 2. Frames (images)

**What.** Single frames of video from **one fixed camera** (the camera must not move or zoom, otherwise the scene
file is no longer valid). JPEG or PNG, any resolution; the paper used 1920 × 1080.

**Frame rate.** Use a constant frame rate and remember it: speeds are computed from the number of frames between
the two counting lines. The paper took every 5th frame of 25 fps video, which gives **5 frames per second**. With
cyclists at 10–40 km/h and lines about 13 m apart, 5 fps gives 6–25 frames per crossing, which works well. Do not
go below about 3 fps.

**Cutting videos into frames.**

```bash
uv run cyclist-tracker dataset extract-frames camera1.mp4 my_dataset/test --every 5
```

or *Preparation tools → Extract frames* in tab 1. Frames are named `<video name>_frame_<number>.jpg`.

**Order and names.** The program processes the frames of a folder in *natural* order of their names
(`frame_9995` comes before `frame_10000`), so the names must sort in time order. Include the frame number in the
name, as the extraction tool does. Keep frames of different videos apart by a different prefix (e.g. date and
time), or analyse one video at a time.

**Consecutive frames.** Tracking, counting and speed need *consecutive* frames of the same video (with a constant
step). For *training* the detector any selection of frames is fine.

**Which frames and how many.** The paper used the busy morning hours (08:00–10:00) of 14 days: 13 877 training,
7 947 validation and 9 241 test frames (96, 64 and 71 cyclists). For fine-tuning a detector on a new camera,
a few thousand annotated frames with some hundreds of different cyclists is a good start. Include difficult
conditions you want the system to handle: rain, low sun, darkness, crowded moments, occlusions.

**Splitting train / val / test.** Split by **day or by video, not by random frames**. Consecutive frames are
nearly identical; if they end up in both train and test, test results look much better than they really are.
The paper assigned whole days to each split (e.g. 5 days train, 3 days validation, 6 days test).

## 3. Annotations (COCO format)

Draw a box around every cyclist in every frame with an annotation tool. We recommend
[CVAT](https://www.cvat.ai/) (free, can be installed locally so images stay private) and its *track* mode, which
interpolates boxes between key frames and gives every cyclist a track ID automatically.

### What to annotate

* **One box per cyclist per frame**, around **rider and bicycle together**, tight to the visible outline.
* Label every box `cyclist`. All boxes in the file are treated as cyclists; do not add other classes to the files
  used for training. Pedestrians, mopeds, scooters and cars are **not** annotated (in the paper, mopeds detected as
  cyclists were the main source of false counts — clear examples without a box teach the model the difference).
* Annotate cyclists that are partly hidden or partly outside the image as far as they are visible.
* Give one cyclist **the same track ID in all frames** in which it is visible; a new cyclist gets a new ID.
  Track IDs are only needed for tracking ground truth, but CVAT track mode adds them for free.
* Annotate all cyclists in the frame, also those outside your counting area, so the detector does not learn
  that such cyclists are "background".
* Frames without any cyclist are allowed but are not used for training.

### Exporting from CVAT

*Actions → Export task dataset → format **COCO 1.0*** (tick *Save images* only if you need them). Rename the file to
`train.json`, `val.json` or `test.json` and put it in `annotations/`.

### File format

A COCO file is a JSON file with three lists:

```json
{
  "images": [
    {"id": 1, "file_name": "test/cam1_20240315_0800_frame_000120.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 0,
     "bbox": [836.42, 703.33, 150.17, 349.71],
     "area": 52516.0, "iscrowd": 0,
     "attributes": {"track_id": 0}}
  ],
  "categories": [{"id": 0, "name": "cyclist"}]
}
```

| Field | Meaning |
|---|---|
| `images[].id` | Unique number of the image within this file |
| `images[].file_name` | Path of the image **relative to the dataset folder** (`test/<name>.jpg`) or just the file name (then it is looked up in the split folder) |
| `annotations[].image_id` | The `id` of the image the box belongs to |
| `annotations[].bbox` | `[x, y, width, height]` in pixels; `x, y` is the **top-left** corner; width and height > 0 |
| `annotations[].attributes.track_id` | The same number for one cyclist in all its frames (CVAT exports this in track mode) |
| `category_id` / `categories` | Not used for training — every box is a cyclist |

If CVAT wrote long paths into `file_name` (e.g. `test/20240315/08/frames/x.jpg`), fix them with:

```bash
uv run cyclist-tracker dataset flatten annotations/test_raw.json annotations/test.json --split test
```

## 4. Ground truth for evaluation (optional)

Ground truth is needed only to measure quality in tab **5 · Evaluate**. It always refers to the frames in `test/`.
Frame numbers in the CSV files count the processed frames **in the order the program reads them**, starting at 0
(so the first frame of `test/` is frame 0). If you analyse only the frames of a frame list, number them within
that list.

### 4.1 `tracking_test.txt` — boxes with cyclist IDs

One line per box, no header:

```
frame_name,track_id,x,y,width,height,confidence,class,visibility
test/cam1_20240315_0800_frame_000120.jpg,1,836.42,703.33,150.17,349.71,1.0,1,1
test/cam1_20240315_0800_frame_000125.jpg,1,851.17,664.93,145.71,329.14,1.0,1,1
```

Only the file name of the first column is used to match frames. Create it from `annotations/test.json`:

```bash
uv run cyclist-tracker dataset coco-to-mot my_dataset/annotations/test.json my_dataset/ground_truth/tracking_test.txt
```

(or *Create tracking ground truth* in tab 1). Track IDs start at 1.

### 4.2 `speeds_test.csv` — reference speeds

```
track_id,speed_kph,direction,crossing_time_sec,frames_taken,entry_frame,exit_frame
1,34.71,down,1.4,7,8,15
3,16.20,up,3.0,15,179,194
```

One row per cyclist that passed both counting lines. `direction` must use the direction names of your scene file.
A predicted speed is paired with a reference speed when the direction is equal and entry and exit frame differ by
at most the *frame tolerance* (default 5 frames).

How to obtain reference speeds:

* **Measured independently** (best): e.g. with a radar gun, or from GPS riders whose passing times you know.
* **Known distance** (as in the paper): the real distance between counting line A and B is measured on the road
  (the paper used the painted blocks: 14 blocks of 0.5 m plus 13 gaps of 0.5 m = 13.5 m) and the speed is that
  distance divided by the time between the lines. With your annotated boxes as perfect detections:

  ```bash
  uv run cyclist-tracker analyze my_dataset/test --scene my_dataset/scene.json \
      --detector annotations --annotations my_dataset/annotations/test.json \
      --speed-mode fixed_distance --output-dir reference --name reference --no-video
  cp reference/reference_speeds.csv my_dataset/ground_truth/speeds_test.csv
  ```

  Check the result by watching some cyclists: the reference is only as good as the measured distance **and the
  moments of entry and exit**. Entry and exit are registered when a cyclist comes within a few pixels of a line
  (15 px, up to 30 px for the exit), i.e. slightly *before* the line. Far from the camera a few pixels are
  metres on the road, so the measured time is too short and this method overestimates speeds. On the demo camera
  (960 × 540 pixels) the overestimate is about 9 km/h; at higher resolution and with lines closer to the camera it
  is smaller. Lower the thresholds in *Advanced settings* when you use this method, or prefer independent
  measurements.

### 4.3 `counts_test.csv` — reference crossings

```
track_id,direction,frames_taken,entry_frame,exit_frame
1,down,8,7,15
3,up,19,93,112
```

One row per cyclist that really rode from one counting line to the other. Count them by watching the test video
(the paper did this manually); a speeds file from 4.2 can be used as a starting point and then corrected.
Cyclists that enter or leave the track between the lines are **not** included.

## 5. The scene file (`scene.json`)

The scene describes one camera position. Make it in tab **2 · Camera scene** by clicking on a camera image;
you rarely need to edit it by hand.

```json
{
  "name": "Camera 1, Main Street",
  "roi": [[937, 390], [1100, 388], [1003, 1080], [469, 1079]],
  "line_a": [[832, 544], [1085, 548]],
  "line_b": [[930, 387], [1103, 387]],
  "direction_a_to_b": "down",
  "direction_b_to_a": "up",
  "calibration_image_points": [[937.3, 390.2], [1100.3, 388.7], [1084.6, 506.2], [862.4, 504.0]],
  "calibration_world_points": [[0, 0], [4, 0], [4, 10.5], [0, 10.5]],
  "reference_distance_m": 13.5
}
```

| Field | How to choose it |
|---|---|
| `roi` | Polygon (pixels) around the cycle track. Detections whose centre lies outside are ignored, so other roads, footpaths and parked bicycles are excluded. |
| `line_a`, `line_b` | Two lines straight across the track, some metres apart (the paper: about 13.5 m). Every cyclist that passes both is counted. Place them where cyclists are well visible and not hidden by poles or trees. |
| `direction_a_to_b`, `direction_b_to_a` | Names written in the results, e.g. `north`/`south` or `to centre`/`from centre`. |
| `calibration_image_points` | At least **4 points on the road surface** (pixels) whose real positions you know, e.g. corners of road markings, tiles or kerb joints. Spread them over the area where speeds are measured; no three on one line. |
| `calibration_world_points` | The same points in **metres**, in any flat coordinate system on the road (x across, y along the track is convenient). Measure with a measuring tape, the design drawings of the road, or a high-resolution aerial photo. |
| `reference_distance_m` | Real distance between line A and line B, used by the *known distance* speed method. |

The tab shows the distance between line A and B *as measured with your calibration*; if this differs clearly
from the real distance, the calibration points or their order are wrong. (For the paper scene: 13.28 m
measured versus 13.5 m painted — 1.6 % difference.)

The calibration assumes a **flat road**. It is valid for points *on* the road, which is why the program follows
the bottom-centre of each box (where the wheels touch the road) by default. Use `--tracking-point center` only to
reproduce the paper exactly.

## 6. Checklist

- [ ] One fixed camera per scene file; frames of one video analysed in one run
- [ ] Constant, known frame rate (5 fps recommended); frame numbers in the file names
- [ ] Frames split into train / val / test **by day or video**
- [ ] Every cyclist boxed in every frame (rider + bicycle), same track ID across frames, no mopeds
- [ ] COCO files in `annotations/`, `file_name` = `<split>/<image>` or `<image>`
- [ ] `scene.json` with region, both lines, direction names and ≥ 4 calibration points
- [ ] Optional ground truth in `ground_truth/` with direction names matching the scene
- [ ] `uv run cyclist-tracker dataset validate my_dataset` shows no red crosses

## 7. Privacy

Camera images of people are personal data. Make sure you are allowed to record and process them (for example
under the GDPR), store them securely, and never commit them to a public repository — `dataset/` and model files
are excluded in `.gitignore`. Everything in Cyclist Tracker runs locally; no images are uploaded.
