# Time series forecasting with Fourier-adjusted time dummies

This is a solution to the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition on Kaggle.

The starting point was provided by the [Darts-based solution by Tom Keldenich](https://inside-machinelearning.com/en/top-1-kaggle-my-method/). Darts is not used here, but a similar logic is applied within a pure Pandas framework.

# Short description

The main focus is pure time series forecasting performance (RMSLE) on the test data set, which was not accessible to me. There is no analysis, or modeling in a traditional sense (like with financial data). I've borrowed ideas from ETS models (error, trend, seasonality) and autoregressive models.

The predictors (features) are engineered in a way similar to models such as AR(p) and Facebook Prophet - but no pre-built models are used here. This is pure machine learning modeling: `LinearRegression()` is used as a "rich baseline". Actual forecasting is done with an ensemble of LightGBM regressors, each trained on slightly different features (different lags).

There is another version of this project, using true ensemble models (linear regressors and random forests, stacked) which, all else being equal, seems to perform better than the single-model "ensembles" used here. I will merge that version here if and when I have time.

Time dummies (actually, one-hot encoded, see article below) are used here to model time-related features. The amplitude of the variation of the time dummies (their envelope) is not fixed (unlike the typical time dummy-based techniques), but it is variable in time. The envelope is extracted from the target via the Fourier spectrogram.

# Detailed description of the Fourier-based technique

More details on how to use the Fourier spectrogram to learn time dummy envelopes can be found in my article on Towards Data Science, which is a companion article for this project:

https://medium.com/data-science/modeling-variable-seasonal-features-with-the-fourier-transform-18c792102047

# Kaggle links

The version of the notebook submitted to Kaggle; it has far fewer models in the ensemble than the code in this repo, in order to speed it up, but performance is slightly worse:

https://www.kaggle.com/code/florinandrei/fourier-spectrogram-ensemble-models

The dataset needed to run the notebook was not included in the repo, because it is too big, but it can be downloaded directly from the competition page:

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
