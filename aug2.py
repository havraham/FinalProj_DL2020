# Importing necessary library
import Augmentor

# Passing the path of the image directory
p = Augmentor.Pipeline("data/train/periodic/")

# Defining augmentation parameters and generating 5 samples
p.flip_left_right(0.5)
p.black_and_white(0.1)
p.rotate(0.3, 10, 10)
p.skew(0.4, 0.5)
p.zoom(probability=0.2, min_factor=1.1, max_factor=1.5)
p.gaussian_distortion(probability=0.2,grid_width=4, grid_height=4, magnitude=8,corner="bell",method="in")
p.greyscale(probability=0.8)
p.process()

p2 = Augmentor.Pipeline("data/train/not_periodic/")

# Defining augmentation parameters and generating 5 samples
p2.flip_left_right(0.5)
p2.black_and_white(0.1)
p2.rotate(0.3, 10, 10)
p2.skew(0.4, 0.5)
p2.zoom(probability=0.2, min_factor=1.1, max_factor=1.5)
p2.gaussian_distortion(probability=0.2,grid_width=4, grid_height=4, magnitude=8,corner="bell",method="in")
p2.greyscale(probability=0.8)
p2.process()