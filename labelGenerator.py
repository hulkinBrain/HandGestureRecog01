from pathlib import Path
import numpy as np
import json

classList = ["one", "two", "three", "four", "five"]
parentFolder = "train"

def perFolderClassLabelGenerator():
    for i in range(len(classList)):
        pathToFolder = Path(f"{parentFolder}/{classList[i]}")
        listOfFilenames = list(pathToFolder.glob('**/*_ann.jpg'))
        listOfFilenames = [annImg.name for annImg in listOfFilenames]

        #### Get number of files in the folder and divide by 2 due to the presence of RGB and Annotated images
        numberOfFiles = int(len(listOfFilenames)/2)

        # Create a vector of length len(classList) and assign 1 to the respective classIndex for an image's class label
        singleImageLabelValue = np.zeros(5, dtype=np.uint8)
        singleImageLabelValue[i] = 1

        # Create the class label vector for each image present in the folder
        imageLabelValues = np.repeat([singleImageLabelValue], numberOfFiles, axis=0)
        np.savetxt(f"{pathToFolder}/labels.csv", imageLabelValues, fmt='%1d')


def singleFolderClassLabelGenerator(createSingleFolder=False):
    imageMap = {}
    for i in range(len(classList)):
        pathToFolder = Path(f"{parentFolder}/{classList[i]}")
        listOfFilenames = list(pathToFolder.glob('**/*_ann.jpg'))
        listOfFilenames = [annImg.name for annImg in listOfFilenames]

        #### Get number of files in the folder and divide by 2 due to the presence of RGB and Annotated images
        numberOfFiles = int(len(listOfFilenames) / 2)

        # Create a vector of length len(classList) and assign 1 to the respective classIndex for an image's class label
        singleImageLabelValue = np.zeros(5, dtype=np.uint8)
        singleImageLabelValue[i] = 1

        for fileName in listOfFilenames:
            imageMap[fileName] = str(singleImageLabelValue)[1:-1].replace(' ', '')

    with open("qwe.txt", "w") as file:
        file.write(json.dumps(imageMap))

singleFolderClassLabelGenerator()


