from pathlib import Path
import numpy as np

classList = ["one", "two", "three", "four", "five"]
parentFolder = "train"
for i in range(len(classList)):
    pathToFolder = Path(f"{parentFolder}/{classList[i]}")
    numberOfFiles = int(len(list(pathToFolder.glob('**/*.jpg')))/2)
    singleImageLabelValue = np.zeros(5, dtype=np.uint8)
    singleImageLabelValue[i] = 1
    imageLabelValues = np.repeat([singleImageLabelValue], numberOfFiles, axis=0)
    np.savetxt(f"{pathToFolder}/labels.csv", imageLabelValues, fmt='%1d')
