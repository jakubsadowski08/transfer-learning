import splitfolders

splitfolders.ratio("merged/train", output="dataset",
                   seed=0, ratio=(.8, .1, .1),
                   group_prefix=None, move=False)
