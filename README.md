# english-dutch-classifier
uses both decision trees and adaptive boosting to classify sentences as English or Dutch </br>

## Usage
The classifier has two learning types: decision tree and ada-boost </br>
You can use either of them for training using the following command.</br>
```
train <training-set-file> <path-to-save-trained-learner> <learning-type> (dt or ada)
```
To evaluate the tree you can use the following command:
```
predict <validation-set-file>
```

## Decision Tree Learner
The decision tree was designed to run recursively like a binary tree because there were only two classes </br>
The max depth of the tree was set to 10. This is to avoid over-fitting the </br>
data but also because the error rate after this depth was not decreasing significantly </br>

testing results: </br>
max-depth = 3; decision tree error rate:  	2.4043715846994536 </br>
max-depth = 7; decision tree error rate:  	1.4207650273224044 </br>
max-depth = 10; decision tree error rate:  	1.3114754098360655 </br>
max-depth = 15; decision tree error rate:  	1.3114754098360655 </br>
decision tree learner error rate on validation data:   2.0


## Ada Boosting
The ada-boosting learner was constructed with the help of the decision tree </br>
algorithm with the max depth set to 1. After testing, the optimal decision stump number turned out to be 50 </br>

testing results: </br>
number of stumps = 100; ada boost error rate:  	2.15633423180593 </br>
number of stumps = 70; ada boost error rate:	2.15633423180593 </br>
number of stumps = 50; ada boost error rate:	2.15633423180593 </br>
ada boost error rate on validation data:	2.888888888888889 </br>
