# SfM

### How to run

1. Generate feature points and matchings
* default feature detector is **SIFT**
* deafult feature matcher is **BFMatcher**
* calibration matrix for data/tiny_duck is **s92**

```
python3 featmatch2.py --data_dir="data/tiny_duck/images" --out_dir="data/tiny_duck" --ext="jpg" --features="SuperPoint" --matcher="FlannBasedMatcher"
```

2. Run SFM to build point clouds
```
python3 sfm.py --calibration_mat="s92" --dataset="tiny_duck" --features="SuperPoint" --ext="jpg" --matcher="FlannBasedMatcher"
```

3. Point clouds are in `results/tiny_duck/point-clouds`