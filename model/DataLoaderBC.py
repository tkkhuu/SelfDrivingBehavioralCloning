import cv2
import numpy as np
from sklearn.utils import shuffle

from TKDNNUtil.DataLoader import DataLoader

class DataLoaderBC(DataLoader):
    def GenerateTrainingBatch(self, samples, batch_size=32, flip_images=True, side_cameras=True):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                car_images = []
                steering_measurements = []
                for line in batch_samples:
                    source_path = line[0]
                    filename = source_path.split('/')[-1]
                    current_path = '../SimData/IMG/' + filename
                    image = cv2.imread(current_path)
                    measurement = float(line[3])
                    car_images.append(image)
                    steering_measurements.append(measurement)
            
                    if flip_images:
                        car_images.append(cv2.flip(image, 1))
                        steering_measurements.append(measurement*-1.0)
                
                    steering_correction = 0.25
                    if side_cameras:
                        left_source_path = line[1]
                        left_filename = left_source_path.split('/')[-1]
                        left_current_path = '../SimData/IMG/' + left_filename
                        left_image = cv2.imread(left_current_path)
                        car_images.append(left_image)
                        steering_measurements.append(measurement + steering_correction)
                        
                        right_source_path = line[2]
                        right_filename = right_source_path.split('/')[-1]
                        right_current_path = '../SimData/IMG/' + right_filename
                        right_image = cv2.imread(right_current_path)
                        car_images.append(right_image)
                        steering_measurements.append(measurement - steering_correction)
                     
                # trim image to only see section with road
                X_train = np.array(car_images)
                y_train = np.array(steering_measurements)
                yield shuffle(X_train, y_train)