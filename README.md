# Stock-Trend-Prediction

SVR with Grid Search was taking way too long for training so I had to terminate it 

Tried using SGD Classifier on the dataset but I am getting some errors. I want to pass onr row from training set to see if the classifier outputs 1 or 0 correctly.

# Ideas about implementing solutions

Can we use price action as our primary idea? 

Price Action is the plot of changes in price of a security over time. All the technical indicators are derived from Price Action itself. 

Candlestick patterns are a way of analysing the price action of a secirity. There are various candlestick patterns which signify a certain change in prices and hint towards a probable direction of the price (All this is speculative, of course). 

A candlestick is generated using the Open, High, Low and Close of a security. This data we already have. So generating candlestick in a numerical way is a matter of deriving new features from existing data. 

But can a model learn to detect these numerical pattern throught the stock price data and understand which pattern lead to what kind of a movement?

A lot of papers have used technical indicators, but I have not come across a paper where pure price action is used. 

Please let me know what you guys think about this.
