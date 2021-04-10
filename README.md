# Stock-Trend-Prediction

SVR with Grid Search was taking way too long for training so I had to terminate it 

Tried using SGD Classifier on the dataset but I am getting some errors. I want to pass onr row from training set to see if the classifier outputs 1 or 0 correctly.

# Ideas about implementing solutions

Can we use price action as our primary idea? 

Price Action is the plot of changes in price of a security over time. All the technical indicators are derived from Price Action itself. 

Candlestick patterns are a way of analysing the price action of a security. There are various candlestick patterns which signify a certain change in prices and hint towards a probable direction of the price (All this is speculative, of course). 

A candlestick is generated using the Open, High, Low and Close of a security. This data we already have. So generating candlestick in a numerical way is a matter of deriving new features from existing data. 

But can a model learn to detect these numerical patterns throught the stock price data and understand which pattern led to what kind of a movement? Detecting these patterns throughout the data and understanding the corresponding movement is the human solution of predictring price direction using Price Action. Is it possible to implement this with a model?

A lot of papers have used technical indicators, but I have not come across a paper where pure price action is used. If you know about  such a study, please let me know.

Please let me know what you guys think about this.


# Preliminary Model Testing

I tried out a few SVMs and a Random Forest model on the data. The models were selected for testing by a cross validation process of rolling window. 
1) Initially 1635 out of the total 2725 samples were sorted out as training samples, 545 as validation samples and 545 as testing samples.
2) The model was trained on 1635 samples and tested on 109 samples from validation set.
3) then the tested 109 samples were added to the training set and the model was trained on the new training set i.e. 1635 + 109 and it was tested on the the samples following the last test samples i.e sample no. 109 to 218 of the validation set
4) This process was continued untill the validation set was exhausted.
5) For every training testing cycle, performance measures were calculated.
6) At the exhaustion of the validation set, i.e. after 5 training testing cycles, the averages of the performance measures were calculated.
7) The model with the best average performance measures was selected for testing. (The files with model names contain the code and the average performance measures).
8) Testing was done on the 545 samplees which were sorted out initially and were never seen by the model. (The file named MODEL TESTING HDFC contains the test results)
9) Normalization of the training, validation and test sets was done separately. 
10) No feature selection was done. All the features were used.
