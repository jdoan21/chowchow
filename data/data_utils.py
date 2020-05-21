from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def download_from_gdrive():
    """Downloads (massive) dataset from user's Google drive"""
    # Authenticate Google Drive account
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    # Download compressed dataset
    download = drive.CreateFile({'id': '1hKFMGIY2jNYbntK4e4aGvwOwvzptP8DH'})
    download.GetContentFile('data.tar.gz')

def load_data(df):
    """Combs through data directories and pixel normalizes data"""
    trainX, testX, valX = [], [], []
    trainY, testY, valY = [], [], []
    
    for i in range(len(df)):
        
        item = df.loc[i][0]
        current_label = np.array((df.loc[i])[1:])
        
        path = os.path.join('images', item)
        list_of_imgs = [os.path.join(path, file) for file in os.listdir(path)]
        train_set = list_of_imgs[:30]
        val_set = list_of_imgs[30:40]
        test_set = list_of_imgs[40:]
        
        for file in train_set:
            img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
            trainX.append(img)
            trainY.append(current_label)
        
        for file in val_set:
            img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
            valX.append(img)
            valY.append(current_label)
        
        for file in test_set:
            img = cv2.resize(cv2.cvtColor(cv2.imread(file, 1), cv2.COLOR_BGR2RGB), (224, 224))
            testX.append(img)
            testY.append(current_label)
    
    # Format arrays
    trainX, trainY = np.array(trainX, dtype=np.float32), np.array(trainY, dtype=np.int32)
    testX, testY = np.array(testX, dtype=np.float32), np.array(testY, dtype=np.int32)
    valX, valY = np.array(valX, dtype=np.float32), np.array(valY, dtype=np.int32)

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = trainX.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = trainX.std(axis=(0, 1, 2), keepdims=True)
    trainX = (trainX - mean_pixel) / std_pixel
    valX = (valX - mean_pixel) / std_pixel
    testX = (testX - mean_pixel) / std_pixel

    return (trainX, trainY, testX, testY, valX, valY)