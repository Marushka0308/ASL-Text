'''import cv2
import cvzone

# Load your pre-trained model and labels
myClassifier = cvzone.Classifier('MyModel/keras_model.h5', 'MyModel/labels.txt')

# Load your test data from a separate file
test_images, test_labels = load_test_data()  # You need to define this function

# Initialize variables to keep track of accuracy
total_test_images = len(test_images)
correct_predictions = 0

for img, true_label in zip(test_images, test_labels):
    # Make predictions using your model
    predictions = myClassifier.getPrediction(img)
    predicted_label = np.argmax(predictions)

    # Compare predictions to the true label
    if predicted_label == true_label:
        correct_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / total_test_images) * 100
print(f"Test Accuracy: {accuracy:.2f}%")'''

'''import cv2
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
myClassifier = Classifier('MyModel/keras_model.h5','MyModel/labels.txt')

while True:
    _,img = cap.read()
    predictions = myClassifier.getPrediction(img)
    cv2.imshow("Image",img)
    cv2.waitKey(1)'''
import cv2
from cvzone.ClassificationModule import Classifier
import time

cap = cv2.VideoCapture(0)
myClassifier = Classifier('MyModel/keras_model.h5', 'MyModel/labels.txt')

# Initialize an empty list to store predictions
predictions_list = []

while True:
    _, img = cap.read()
    predictions = myClassifier.getPrediction(img)

    # Append the prediction to the list
    predictions_list.append(predictions)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    time.sleep(2)


    # Break the loop if you want to display a certain number of predictions
    # For example, if you want to display the last 10 predictions:
    if len(predictions_list) >= 20:
        break

# Display the list of predictions
#print("List of Predictions:", predictions_list)
string_list = []
pred = []
for i in range(0,len(predictions_list)):
    #print(predictions_list[i][1])
    pred.append(predictions_list[i][1])
for i in pred:
    if i==0:
        string_list.append('A')
    elif i==1:
        string_list.append('B')
    elif i==2:
        string_list.append(' ')
print(string_list)
# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()




