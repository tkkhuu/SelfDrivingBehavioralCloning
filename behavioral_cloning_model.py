import DNNBuilder
import DataLoader

from sklearn.model_selection import train_test_split

def normalize_image(input_img):
    return (input_img / 255.0) - 0.5

# Loading data
data_loader = DataLoader.DataLoader('../SimData/driving_log.csv')
data = data_loader.LoadData()

# Splitting train and validation data
train_samples, validation_samples = train_test_split(data, test_size=0.2)

flip_images = False
side_cameras = True

BATCH_SIZE = 16
EPOCHS = 8

train_generator = data_loader.GenerateTrainingBatch(train_samples, batch_size=BATCH_SIZE, flip_images=flip_images)
validation_generator = data_loader.GenerateTrainingBatch(validation_samples, batch_size=BATCH_SIZE, flip_images=flip_images)

n_train = len(train_samples)
n_valid = len(validation_samples)

if side_cameras and flip_images:
    n_train *= 4
    n_valid *= 4
elif side_cameras:
    n_train *= 3
    n_valid *= 3
elif flip_images:
    n_train *= 2
    n_valid *= 2

print(n_train, n_valid)

# Defining layers
layers = (
                {'layer_type': 'cropping', 'crop_dim': ((60, 20), (0, 0))},
                {'layer_type': 'lambda', 'function': normalize_image},
                {'layer_type': 'convolution', 'shape': (5, 5, 25), 'stride': (2, 2), 'activation': 'relu'},
                {'layer_type': 'convolution', 'shape': (2, 2, 36), 'stride': (2, 2), 'activation': 'relu'},
                {'layer_type': 'convolution', 'shape': (3, 3, 48), 'stride': (2, 2), 'activation': 'relu'},
                {'layer_type': 'convolution', 'shape': (3, 3, 64), 'activation': 'relu'},
                {'layer_type': 'dropout', 'keep_prob': 0.4},
                {'layer_type': 'flatten'},
                {'layer_type': 'dropout', 'keep_prob': 0.5},
                {'layer_type': 'fully connected', 'output_dim': 100, 'activation': 'relu'},
                {'layer_type': 'fully connected', 'output_dim': 50, 'activation': 'relu'},
                {'layer_type': 'fully connected', 'output_dim': 10, 'activation': 'relu'},
                {'layer_type': 'fully connected', 'output_dim': 1}
        )
    
    
mydnn = DNNBuilder.DNNSequentialModelBuilder2D({'input_shape': (160, 320, 3), 'model_architecture': layers})
mydnn.Initialize()
mydnn.Compile('mse', 'adam')
print(mydnn)
mydnn.TrainAndSave(train_generator, validation_generator, n_train, n_valid, EPOCHS)