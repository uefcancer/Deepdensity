import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("dataset/", output="dateset_name",
    seed=1, ratio=(.70, .15, .15), group_prefix=None, move=False) # default values