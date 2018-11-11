# HOGNeuralNet
Creation of a neural network to classify HOG data gathered from the Jaffe database.

This nerual network has a single hidden layer which can use a number of activation functions. The output layer consists of 7 neruons using a sigmoid function for an activation. The final output uses the one-hot method, with the maximum value in the set becoming the 1 value, and the rest being reduced to 0. 

# Emotion Labels 
1000000 Angry
0100000 Disgust
0010000 Fear
0001000 Happy
0000100 Neutral
0000010 Sad
0000001 Surprise

# Program Output
```
Program Starting

Read in 213 images from the Jaffe database.

	 30 	 Angry
	 29 	 Disgust
	 32 	 Fear
	 31 	 Happy
	 30 	 Neutral
	 31 	 Sad
	 30 	 Surprise
	---------------------
	 159 	 Training Examples
	 54 	 Testing Examples

Running HOG
Training Neural Network
	---
Cost at 0 4.851773795072829
Accuracy: 13.20754716981132
Cost at 50 2.770205801525372
Accuracy: 30.81761006289308
Cost at 100 2.018388789910542
Accuracy: 66.0377358490566
Cost at 150 1.7183135887095513
Accuracy: 63.52201257861635
Cost at 200 0.5222544032112558
Accuracy: 96.22641509433963
Cost at 250 0.1749239751147752
Accuracy: 99.37106918238993
Cost at 300 0.07886075520084389
Accuracy: 100.0
Cost at 350 0.05115073052317008
Accuracy: 100.0
Cost at 400 0.03672703543742705
Accuracy: 100.0
Cost at 450 0.028140808053845397
Accuracy: 100.0
Cost at 500 0.02254917335028536
Accuracy: 100.0
Cost at 550 0.018662424354947645
Accuracy: 100.0
Cost at 600 0.01582855457931452
Accuracy: 100.0
Cost at 650 0.013683022699696836
Accuracy: 100.0
Cost at 700 0.012009713185312425
Accuracy: 100.0
Cost at 750 0.010673442543944927
Accuracy: 100.0
Cost at 800 0.009583930287963505
Accuracy: 100.0
Cost at 850 0.008680526551903303
Accuracy: 100.0
	---
Making Predictions

Accuracy: 90.74 %


Times:
	 0.10809 	Read In Time
	 19.64181 	HOG Transformation Time
	 182.11333 	Training Time
	 0.05475 	Training Time
	---------------------
	 201.91798 	Total Time
```
