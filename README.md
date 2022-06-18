# SA-b-PET-ANIMAL-CLASSIFIER

# Algorithm
 
1.Import libraries required.

2.Load dataset through local or drive link.

3.Train the datasets.

4.Train the model with neural networks.

5.Compile the code.

6.Fit the model and check accuracy.

7.Evaluate performance on test dataset.

## Program:

Program to implement 
Developed by   : Pritikha A P
RegisterNumber :  212219040114

1.code

```python3
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
X_train = np.loadtxt('input.csv', delimiter = ',')
Y_train = np.loadtxt('labels.csv', delimiter = ',')
X_test = np.loadtxt('input_test.csv', delimiter = ',')
Y_test = np.loadtxt('labels_test.csv', delimiter = ',')
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)
X_train = X_train/255.0
X_test = X_test/255.0
print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 5, batch_size = 64)
model.evaluate(X_test, Y_test)
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()
y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5
if(y_pred == 0):
    pred = 'cat'
else:
    pred = 'dog'
    
print("Our model says it is a :", pred)
```
2. DEMO VIDEO YOUTUBE LINK:

## OUTPUT:
![173228395-d1f676ca-0a6e-4854-ab98-d78986276c90](https://user-images.githubusercontent.com/78891075/174431133-7e08da4e-9708-4532-a3e1-b4ba12b2a3a4.png)

![173228407-f38ee8bb-f516-4985-ba7f-4fbd0c1f0edf](https://user-images.githubusercontent.com/78891075/174431138-48719433-f4fe-4c0f-a0ab-7cd321eb3755.png)


![173228418-7d4d8c02-b4f3-4f8f-8965-ef23cc53a08e](https://user-images.githubusercontent.com/78891075/174431142-48cf36ec-eeda-4f78-afca-e7240e679242.png)


![173228429-413de8c1-b7bc-4c7d-aa72-1b62ed0fec3b](https://user-images.githubusercontent.com/78891075/174431147-2bbd91e5-ba8a-4632-95ad-86dadd5e22ec.png)
