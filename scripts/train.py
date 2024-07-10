import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import time


from scripts.transform import TransformData

# ML models and utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score



# tpr (True Positive Rate) vs. fpr (False Positive Rate) dataframe
    # tp = True Positive
    # tn = True Negative
    # fp = False Positive
    # fn = False Negative
    # Decision Rule :  "y_pred>= Threshold" for Class "1"

    # when only_even=True --> we'll have a step ==0.02 and leave only even records
def tpr_fpr_dataframe(y_true, y_pred, only_even=False):
    scores = []

    if only_even==False:
        thresholds = np.linspace(0, 1, 101) #[0, 0.01, 0.02, ...0.99,1.0]
    else:
        thresholds = np.linspace(0, 1, 51) #[0, 0.02, 0.04,  ...0.98,1.0]

    for t in thresholds:

        actual_positive = (y_true == 1)
        actual_negative = (y_true == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        if tp + fp > 0:
            precision = tp / (tp + fp)

        if tp + fn > 0:
            recall = tp / (tp + fn)

        if precision+recall > 0:
            f1_score = 2*precision*recall / (precision+recall)

        accuracy = (tp+tn) / (tp+tn+fp+fn)

        scores.append((t, tp, fp, fn, tn, precision, recall, accuracy, f1_score))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn','precision','recall', 'accuracy','f1_score']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)

    return df_scores

# Function to find all predictions (starting from 'pred'), generate is_correct (correctness of each prediction)
# and precision on TEST dataset (assuming there is df["split"] column with values 'train','validation','test'

# returns 2 lists of features: PREDICTIONS and IS_CORRECT
def get_predictions_correctness(df:pd.DataFrame, to_predict:str):
    PREDICTIONS = [k for k in df.keys() if k.startswith('pred')]
    print(f'Prediction columns founded: {PREDICTIONS}')

    # add columns is_correct_
    for pred in PREDICTIONS:
        part1 = pred.split('_')[0] # first prefix before '_'
        df[f'is_correct_{part1}'] =  (df[pred] == df[to_predict]).astype(int)

    # IS_CORRECT features set
    IS_CORRECT =  [k for k in df.keys() if k.startswith('is_correct_')]
    print(f'Created columns is_correct: {IS_CORRECT}')

    print('Precision on TEST set for each prediction:')
    # define "Precision" for ALL predictions on a Test dataset (~4 last years of trading)
    for i,column in enumerate(IS_CORRECT):
        prediction_column = PREDICTIONS[i]
        is_correct_column = column
        filter = (df.split=='test') & (df[prediction_column]==1)
        print(f'Prediction column:{prediction_column} , is_correct_column: {is_correct_column}')
        print(df[filter][is_correct_column].value_counts())
        print(df[filter][is_correct_column].value_counts()/len(df[filter]))
        print('---------')

    return PREDICTIONS, IS_CORRECT

# See https://blog.csdn.net/Gou_Hailong/article/details/129778588
# Basically, this is a Binary Search Algorithm
def lower_bound(values, threshold):
    low, high = 0, len(values) - 1

    while low <= high:
        mid = low + (high - low) // 2
        if values[mid] < threshold:
            low = mid + 1
        else:
            high = mid - 1
    
    return low


class TrainModel:
    transformed_df: pd.DataFrame #input dataframe from the Transformed piece 
    df_full: pd.DataFrame #full dataframe with DUMMIES

    # Dataframes for ML
    train_df:pd.DataFrame
    test_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_valid_df:pd.DataFrame

    X_train:pd.DataFrame
    X_valid:pd.DataFrame
    X_test:pd.DataFrame
    X_train_valid:pd.DataFrame
    X_all:pd.DataFrame

    # feature sets
    GROWTH: list
    OHLCV: list
    CATEGORICAL: list
    TO_PREDICT: list
    TECHNICAL_INDICATORS: list 
    TECHNICAL_PATTERNS: list
    MACRO: list
    NUMERICAL: list
    CUSTOM_NUMERICAL: list
    DUMMIES: list


    def __init__(self, transformed:TransformData):
        # init transformed_df
        self.transformed_df = transformed.transformed_df.copy(deep=True)
        self.transformed_df['ln_volume'] = self.transformed_df.Volume.apply(lambda x: np.log(x) if x >0 else np.nan)
        # self.transformed_df['Date'] = pd.to_datetime(self.transformed_df['Date']).dt.strftime('%Y-%m-%d')

    def _define_feature_sets(self):
        self.GROWTH = [g for g in self.transformed_df if (g.find('growth_')==0)&(g.find('future')<0)]
        self.OHLCV = ['Open','High','Low','Close','Adj Close','Volume']
        self.CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
        self.TO_PREDICT = [g for g in self.transformed_df.keys() if (g.find('future')>=0)]
        self.MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
            'DGS1', 'DGS5', 'DGS10']
        self.CUSTOM_NUMERICAL = ['vix_adj_close','SMA10', 'SMA20', 'SMA50', 'growing_moving_average', 'declining_moving_average',
                                 'high_minus_low_relative','volatility', 'ln_volume']
        
        # artifacts from joins and/or unused original vars
        self.TO_DROP = ['Year','Date','Month_x', 'Month_y', 'index', 'Quarter','index_x','index_y'] + self.CATEGORICAL + self.OHLCV

        # All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
        self.TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
        'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
        'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
        'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
        'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
        'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
        'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
        'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
        'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice', 'BBU', 'BBM', 'BBL',
        'CVR3VIX_buy', 'CVR3VIX_sell']
        self.TECHNICAL_PATTERNS =  [g for g in self.transformed_df.keys() if g.find('cdl')>=0]
        
        self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + \
            self.CUSTOM_NUMERICAL + self.MACRO
        
        # CHECK: NO OTHER INDICATORS LEFT
        self.OTHER = [k for k in self.transformed_df.keys() if k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT]
        print(f"Others: {self.OTHER}")
        return
  
  
    def _define_dummies(self):
        """Generate dummy variables."""
        # https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/decisionpoint-trend-model
        self.transformed_df['DPTrend_buy'] = (self.transformed_df['SMA10'] > self.transformed_df['SMA20'])
        self.transformed_df['DPTrend_sell'] = (self.transformed_df['SMA10'] < self.transformed_df['SMA20']) & \
                                              (self.transformed_df['SMA20'] < self.transformed_df['SMA50'])
        self.transformed_df['DPTrend_neutral'] = (self.transformed_df['SMA10'] < self.transformed_df['SMA20']) & \
                                                 (self.transformed_df['SMA20'] > self.transformed_df['SMA50'])
        
        # https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/bollinger-band-squeeze
        self.transformed_df['BBand_Squeeze'] = (self.transformed_df['Volume'].rolling(20).mean() > 100000) & \
                                               (self.transformed_df['Close'].rolling(60).mean() > 20) & \
                                               ((self.transformed_df['BBU'] - self.transformed_df['BBL']) / self.transformed_df['Close'] < 0.04)
        
        knowledge_dummies = ['DPTrend_buy', 'DPTrend_sell', 'DPTrend_neutral', 'BBand_Squeeze']
        # Risk Aversion with utility U(r) = E(r) - 0.5*A*Var(r), where 
        # r isthe return  represented by decimal numbers and A > 0
        return_1d = self.transformed_df['growth_1d']-1
        E_r = return_1d.rolling(10).mean()    # expected return
        Var_r = return_1d.rolling(10).var()   # variance of return
        for i in range(1, 11):
            A = i * 0.5   # index of the investor’s risk aversion
            dummy = f'RiskAversion_{A}'
            self.transformed_df[dummy] = (E_r - 0.5 * A *Var_r > 0)
            knowledge_dummies.append(dummy)
        
        self.df_full = self.transformed_df.copy()
        self.DUMMIES = knowledge_dummies
  
    def _perform_temporal_split(self, df:pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
        """
        Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

        Args:
            df (DataFrame): The DataFrame to split.
            min_date (str or Timestamp): Minimum date in the DataFrame.
            max_date (str or Timestamp): Maximum date in the DataFrame.
            train_prop (float): Proportion of data for training set (default: 0.7).
            val_prop (float): Proportion of data for validation set (default: 0.15).
            test_prop (float): Proportion of data for test set (default: 0.15).

        Returns:
            DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
        """
        # Define the date intervals
        train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
        val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

        # Assign split labels based on date ranges
        split_labels = []
        for date in df['Date']:
            if date <= train_end:
                split_labels.append('train')
            elif date <= val_end:
                split_labels.append('validation')
            else:
                split_labels.append('test')

        # Add 'split' column to the DataFrame
        df['split'] = split_labels

        return df
  
    def _define_dataframes_for_ML(self, features_list, to_predict='is_positive_growth_5d_future'):
        #features_list = self.NUMERICAL+ self.DUMMIES
        # What we're trying to predict?
        #to_predict = 'is_positive_growth_5d_future'

        self.train_df = self.df_full[self.df_full.split.isin(['train'])].copy(deep=True)
        self.valid_df = self.df_full[self.df_full.split.isin(['validation'])].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full.split.isin(['train','validation'])].copy(deep=True)
        self.test_df =  self.df_full[self.df_full.split.isin(['test'])].copy(deep=True)

        # Separate numerical features and target variable for training and testing sets
        self.X_train = self.train_df[features_list+[to_predict]]
        self.X_valid = self.valid_df[features_list+[to_predict]]
        self.X_train_valid = self.train_valid_df[features_list+[to_predict]]
        self.X_test = self.test_df[features_list+[to_predict]]
        # this to be used for predictions and join to the original dataframe new_df
        self.X_all =  self.df_full[features_list+[to_predict]].copy(deep=True)

        # Clean from +-inf and NaNs:

        self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
        self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
        self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
        self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
        self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)


        self.y_train = self.X_train[to_predict]
        self.y_valid = self.X_valid[to_predict]
        self.y_train_valid = self.X_train_valid[to_predict]
        self.y_test = self.X_test[to_predict]
        self.y_all =  self.X_all[to_predict]

        # remove y_train, y_test from X_ dataframes
        del self.X_train[to_predict]
        del self.X_valid[to_predict]
        del self.X_train_valid[to_predict]
        del self.X_test[to_predict]
        del self.X_all[to_predict]
        
        print("Current X_train, X_validation, X_test, and X_all shapes:")
        print(f'length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}')
        print(f'  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}')

    def _clean_dataframe_from_inf_and_nan(self, df:pd.DataFrame):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df 
    def prepare_dataframe(self):
        print("Prepare the dataframe: define feature sets, add dummies, temporal split")
        self._define_feature_sets()
        # get dummies and df_full
        self._define_dummies()
        
        # temporal split
        min_date_df = self.df_full.Date.min()
        max_date_df = self.df_full.Date.max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        # define dataframes for ML
        self._define_dataframes_for_ML(features_list=self.NUMERICAL+ self.DUMMIES)

        return 
    
    # Reference: https://builtin.com/machine-learning/pca-in-python
    def _generate_PCA_feature(self):
        """Perform feature reduction using PCA"""
        print("-- Step 3.1: Use PCA to generate an additional feature")
        scaler = StandardScaler()
        scaler.fit(self.X_train_valid)

        # transform into scaled train_validate set and test set
        X_train_valid = scaler.transform(self.X_train_valid)
        X_test = scaler.transform(self.X_test)
        X_all = scaler.transform(self.X_all)

        # try to find the number of features used in PCA
        pca = PCA(n_components='mle')
        pca.fit(X_train_valid)
        cum_ratio = np.cumsum(pca.explained_variance_ratio_)

        # find the index for which the cumulative explained ratio is at least 85%
        # add 1 because the index is less one than the number of elements from the beginning
        n_comp = lower_bound(cum_ratio, 0.85) + 1  
        print(f"For cummulative explained ratio at least 85%, \
index: {n_comp-1}, value: {cum_ratio[n_comp-1]}")

        # retrain the PCA
        pca = PCA(n_components=n_comp)
        pca.fit(X_train_valid)
        X_train_valid_pca = pca.transform(X_train_valid)
        X_test_pca = pca.transform(X_test)
        X_all_pca = pca.transform(X_all)

        logisticReg = LogisticRegression(solver="lbfgs")
        logisticReg.fit(X_train_valid_pca, self.y_train_valid)
        y_test_pred = logisticReg.predict(X_test_pca)
        #print(np.array(np.unique(y_test_pred, return_counts=True)).T)
        #print(tpr_fpr_dataframe(self.y_test, y_test_pred, only_even=True))

        y_pred = logisticReg.predict(X_all_pca)
        return y_pred, n_comp
    
    def _feature_selection(self):
        """Use LogisticRegression and transformer to select important features"""
        print("-- Step 3.2: Use the Logistic Regression and transformer to select important features")
        scaler = StandardScaler()
        scaler.fit(self.X_train_valid)

        # transform into scaled train_validate set and test set
        X_train_valid = scaler.transform(self.X_train_valid)
        X_test = scaler.transform(self.X_test)
        #X_all = scaler.transform(self.X_all)

        # See : https://scikit-learn.org/stable/modules/feature_selection.html and 
        # Or : https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
        # Other approaches: https://medium.com/@rithpansanga/logistic-regression-for-feature-selection-selecting-the-right-features-for-your-model-410ca093c5e0
        lgsr = LogisticRegression(penalty="l2", C=0.05, solver="newton-cholesky")
        model = SelectFromModel(estimator=lgsr, threshold="0.5*mean")
        model.fit(X_train_valid, self.y_train_valid)

        print("Save and return feature list...")
        feature_list = list(self.X_all.columns[model.get_support()]) # remember to convert the result into type list
        self._save_feature_list(feature_list)
        return feature_list
    
    def _tuning_random_forest(self):
        """Tuning the hyperparameters of the Random Forest"""
        print("---- Step 3.3.1: Hyperparameters tuning for the random forest...")
        precision_matrix = {}
        best_precision = 0
        best_depth = 0
        best_estimators = 1

        for depth in [15, 16, 17, 18, 19, 20]:
            for estimators in [50,100,200,500]:
                print(f'Working with HyperParams: depth = {depth}, estimators = {estimators}')
                
                # Start timing
                start_time = time.time()
                # fitting the tree on X_train, y_train
                rf = RandomForestClassifier(n_estimators = estimators,
                                            max_depth = depth,
                                            random_state = 42,
                                            n_jobs = -1)
                
                rf = rf.fit(self.X_train_valid, self.y_train_valid)
                
                # getting the predictions for TEST and accuracy score
                y_pred_valid = rf.predict(self.X_valid)
                precision_valid = precision_score(self.y_valid, y_pred_valid)
                y_pred_test = rf.predict(self.X_test)
                precision_test = precision_score(self.y_test, y_pred_test)
                
                print(f'  Precision on test is {precision_test}, (precision on valid is {precision_valid} - tend to overfit)')\
                
                # saving to the dict
                # 
                precision_matrix[depth, estimators] = round(precision_test,4)
                
                # Measure elapsed time
                elapsed_time = time.time() - start_time
                print(f'Time for training: {elapsed_time:.2f} seconds, or {elapsed_time/60:.2f} minutes')
                
                # updating the best precision
                if precision_test >= best_precision:
                    best_precision = round(precision_test,4)
                    best_depth = depth
                    best_estimators = estimators
                    print(f'New best precision found for depth={depth}, estimators = {estimators}')
                
                print('------------------------------')
        
        print(f'Matrix of precisions: {precision_matrix}')
        print(f'The best precision is {best_precision} and the best depth is {best_depth} ')

        return precision_matrix, best_depth, best_estimators
    
    def _save_rfbest_parameters(self, parameters: dict):
        """Save the best result of Random Forest Parameters"""
        data_dir = "local_data/"
        os.makedirs(data_dir, exist_ok=True)

        file_name = 'rfbest_parameters.joblib'
        if os.path.exists(file_name):
            os.remove(file_name)
        joblib.dump(parameters, os.path.join(data_dir,file_name))

    def _load_rfbest_parameters(self):
        """Load the best result of Random Forest Parameters from the local directory"""
        data_dir = "local_data/"
        os.makedirs(data_dir, exist_ok=True)

        return joblib.load(os.path.join(data_dir,'rfbest_parameters.joblib'))
    
    def _save_feature_list(self, feature_list: list[str]):
        """Save the feature list generated by Logistic Regression and Transformer"""
        data_dir = "local_data/"
        os.makedirs(data_dir, exist_ok=True)

        file_name = 'feature_list.joblib'
        if os.path.exists(file_name):
            os.remove(file_name)
        
        joblib.dump(feature_list, os.path.join(data_dir,file_name))
    
    def _load_feature_list(self):
        """Load the feature list generated by Logistic Regression and Transformer previously"""
        data_dir = "local_data/"
        os.makedirs(data_dir, exist_ok=True)

        return joblib.load(os.path.join(data_dir,'feature_list.joblib'))
  
    def train_model(self, train_new=True, tuning_rf=True):
        """
        Get the result of our designed model
        if train_new is True, it will train a new Random Forest model
        if tuning_rf is True, it will perform hyperparameters tuning for the new Random Forest model
        if train_new is False, then it will load the previous training result and ignore tuning_rf
        """
        pca_feature, n_comp = self._generate_PCA_feature()
        
        # add the PCA feature to the original dataframe and update the dataframes for ML
        self.df_full['PCA_feature'] = pca_feature
        self.DUMMIES.append('PCA_feature')
        self._define_dataframes_for_ML(features_list=self.NUMERICAL+ self.DUMMIES)

        if train_new:
            features = self._feature_selection()
            print(f"features({len(features)} total):")
            for feature in features:
                print(feature)
            

            print("-- Step 3.3: Train a Random Forest model")
            self._define_dataframes_for_ML(features_list=features)

            if tuning_rf:
                precision_matrix, best_depth, best_estimators = self._tuning_random_forest()
                #print(precision_matrix)
                #print(best_depth, best_estimators)
                # save parameters
                self._save_rfbest_parameters({
                        'precision_matrix': precision_matrix,
                        'best_depth': best_depth,
                        'best_estimators': best_estimators
                })
            else:
                # load parameters
                parameters = self._load_rfbest_parameters()
                precision_matrix = parameters['precision_matrix']
                best_depth = parameters['best_depth']
                best_estimators = parameters['best_estimators']
            
            # https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles
            print(f'Training the best model (RandomForest (max_depth={best_depth}, n_estimators={best_estimators}))')
            model = RandomForestClassifier(n_estimators=best_estimators, 
                                                max_depth=best_depth,
                                                # random_state=42,
                                                n_jobs=-1)
            
            self.model = model.fit(self.X_train_valid, self.y_train_valid)
            self.persist("local_data/")
        else:
            print("-- Step 3.3: Load the previous trained Random Forest model")
            features = self._load_feature_list()
            self._define_dataframes_for_ML(features_list=features)
            self.load("local_data/")
        
        print("-- Step 3.4: Add new predictors generated from the trained Random Forest")
        y_pred_all = self.model.predict_proba(self.X_all)
        y_pred_all_class1 = np.array([k[1] for k in y_pred_all])
        self.df_full['proba_pred_class1'] = y_pred_all_class1

        for percent in range(51, 61):
            threshold = percent / 100
            predictor_name = f"pred{percent-50}_rf_best_positive_rule{percent}"
            self.df_full[predictor_name] = (y_pred_all_class1 >= threshold)
        
        
  
    def persist(self, data_dir:str):
        '''Save dataframes to files in a local directory 'dir' '''
        os.makedirs(data_dir, exist_ok=True)      

        # Save the model to a file
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir,model_filename)
        joblib.dump(self.model, path)

    def load(self, data_dir:str):
        """Load files from the local directory"""
        os.makedirs(data_dir, exist_ok=True)   
        
        # Load the model to a file
        model_filename = 'random_forest_model.joblib'
        path = os.path.join(data_dir,model_filename)
        self.model  = joblib.load(path)

    def make_inference(self, pred_name:str="my_model"):
        print('-- Step 3.5: Making inference')
        y_test_pred = self.model.predict_proba(self.X_test)
        y_test_pred_df = pd.DataFrame(y_test_pred, columns=["Class0_probability", "Class1_probability"])
        
        print("Statistics Summary:")
        print(y_test_pred_df.describe().T)

        print("Unconditional probability for is_positive_growth: ", \
                               self.y_test.sum() / self.y_test.count())
         
        print("tpr/fpr scores for is_positive_growth_5d_future = 0:")
        df_scores_class0 = tpr_fpr_dataframe(self.y_test, y_test_pred_df["Class0_probability"])
        print(df_scores_class0[(df_scores_class0.threshold>=0.5) & (df_scores_class0.threshold<=0.8)])

        print("tpr/fpr scores for is_positive_growth_5d_future = 1:")
        df_scores_class1 = tpr_fpr_dataframe(self.y_test, y_test_pred_df["Class1_probability"])
        print(df_scores_class1[(df_scores_class1.threshold>=0.5) & (df_scores_class1.threshold<=0.8)])

        print("The graph: ")
        fig, ax = plt.subplots(2, 1)
        df_scores_class0.plot.line(
            x='threshold', 
            y=['precision','recall', 'f1_score'],
            title = 'Precision vs. Recall for the Best Model (is_positive_growth_5d_future = 0)',
            ax=ax[0])
        df_scores_class1.plot.line(
            x='threshold', 
            y=['precision','recall', 'f1_score'],
            title = 'Precision vs. Recall for the Best Model (is_positive_growth_5d_future = 1)',
            ax=ax[1])
        
        plt.subplots_adjust(hspace=0.5)
        plt.show()

        predictions, is_correct = get_predictions_correctness(self.df_full, to_predict='is_positive_growth_5d_future')