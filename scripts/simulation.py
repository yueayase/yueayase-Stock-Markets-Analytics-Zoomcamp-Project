import numpy as np
import pandas as pd
from scipy import optimize


from scripts.train import TrainModel

class Simulation:
    train_model: TrainModel
    rf_best_pred: str
    invest_each: float

    def __init__(self, train_model: TrainModel, invest_each: float):
        rf_best_pred = ""
        rf_best_precision = 0.0
        PREDICTIONS = [k for k in train_model.df_full.keys() if k.startswith('pred')]
        IS_CORRECT =  [k for k in train_model.df_full.keys() if k.startswith('is_correct_')]

        for i,column in enumerate(IS_CORRECT):
            prediction_column = PREDICTIONS[i]
            is_correct_column = column
            filter = (train_model.df_full.split=='test') & (train_model.df_full[prediction_column]==1)
            current_positive = train_model.df_full[filter][is_correct_column].value_counts().iloc[0]
            current_total = len(train_model.df_full[filter])
            current_precision = current_positive / current_total

            if current_precision > rf_best_precision:
                rf_best_precision = current_precision
                rf_best_pred = prediction_column
        
        

        self.train_model = train_model
        self.rf_best_pred = rf_best_pred
        self.invest_each = invest_each

        # get future 1d growth for optimization methods
        new_df = self.train_model.df_full.copy()
        for day in range(1, 5):
            new_df[f'growth_future_{day}d'] = new_df['Adj Close'].shift(-day) / new_df['Adj Close']
        self.train_model.df_full = new_df

        # optimization methods
        self.opt_methods = {
            "EW": self.equal_weighting,
            "MVO": self.mean_variance_optimization,
            "SRO": self.sharpe_ratio_optimization
        }

    def report(self, title: str, opt_method: str):
        print(title)
        df = self.train_model.df_full
        pred_rank_column_name = self.rf_best_pred + "_rank"
        # get the ranks of each predicted probablity for each day 
        df[pred_rank_column_name] = df.groupby("Date")["proba_pred_class1"].rank(method="first", ascending=False)
        DATES = df[df.split == "test"].sort_values(by="Date").Date.unique()  # get the all trading days for the TEST set
        

        gross_rev = 0.0        # gross revenue
        total_fees = 0.0             # total fees
        count_investments = 0  # how many investments?
        total_capital = 0.0
        for date in DATES[:-5]:
            # get the most probable growing top 5 stocks
            one_day_pred_top5 = df[(df.Date == date) & (df[self.rf_best_pred] == 1) & (df[pred_rank_column_name] <= 5)]
            no_of_buy = len(one_day_pred_top5)   # total number of possible candidates
            if no_of_buy > 0:  # have trade
                # if there is nan in the growth_future_5d, you only invest the money but won't sell that stock after 5 days
                # So, in this case, the growth rate = -100%, ie. you cost the money you've invest
                # Ohterwise, the growth rate = row["growth_future_5d"]-1
                new_growth = one_day_pred_top5.apply(
                    lambda row: -1 if np.isnan(row["growth_future_5d"]) else row["growth_future_5d"]-1,
                    axis=1
                ) 

                invest_weight =self.opt_methods[opt_method](one_day_pred_top5)   # get weights

                # total revenue in this investment
                # The overall growth rate is just the inner product of the weights for all the stocks you bought 
                # and the corresponind rate of growth. 
                # The growth multiplied by the total money you've invest is the revenue of this investment
                gross_rev += (invest_weight @ new_growth) * self.invest_each 

                # Each fee should be self.invest_each / no_of_buy * 0.002 and there are no_of_buy for each buy/sell
                # So, you have to pay self.invest_each / o_of_buy * 0.002 * no_of_buy = self.invest_each * 0.002 for fees
                total_fees -= self.invest_each * 0.002
                count_investments += no_of_buy
                total_capital += self.invest_each
        
        net_rev = gross_rev + total_fees        # net revenue = the total revenue you got - the totol fees
        total_days = (DATES[-6]-DATES[0]).days
        CAGR = (1 + net_rev/total_capital)**(1/float(total_days/365))

        print('=========================================================================+')
        print(f"Trading dates: from {DATES[0]} to {DATES[-1]}")
        print(f"There are {count_investments} investments out of {len(df[df.split == 'test'])} TEST records")
        print("Financial Result:")
        print(f"     Gross Revenue: ${np.round(gross_rev, 6)}")
        print(f"     Fees (0.2% for buy/sell): ${np.round(-total_fees, 6)}")
        print(f"     Net Revenue: ${np.round(net_rev, 6)}")
        print(f"     Fees are {np.round(100*(-total_fees)/gross_rev, 2)}% from Gross Revenue")
        print(f"     Capital Required: ${total_capital}")
        print(f"     Final Capital: ${np.round(total_capital+net_rev, 6)}")
        print(f"     Average CVGR on TEST: {np.round(100*(CAGR-1), 3)}%")
        print(f"     Average daily stats:")
        print(f"         Average net revenue per investment: ${np.round(net_rev/count_investments, 6)}")
        print(f"         Average investments per day: {np.round(count_investments/total_days, 6)} ")
        print('=========================================================================+')

    # Simulation Methods
    # Here, you can reference the following 3 links to understand the detailed reasons of my implementation
    # They also provide some concepts of portfolio optimization
    # 1. https://medium.com/@phindulo60/portfolio-optimization-with-python-mean-variance-optimization-mvo-and-markowitzs-efficient-64acb3b61ef6
    # 2. https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios
    # 3. https://www.kaggle.com/code/vijipai/lesson-6-sharpe-ratio-based-portfolio-optimization
    
    # return the weights with equal values
    def equal_weighting(self, one_day_pred_top5: pd.DataFrame):
        no_of_buy = len(one_day_pred_top5)
        return [1 / no_of_buy] * no_of_buy
    
    # helper functions for maximize returns and minimize risk, the approach is from the following link:
    # https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios
    def _maximize_returns(self, mean_returns):
        # https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html#optimize-linprog-highs
        # Note: 
        # 1. c have to take negative because the function we'll use will try to minimize the object function
        # 2. A must be a 2D array, so we can't write A = np.ones([len(mean_returns)])
        # 3. b must be a 1D array, so we can't write b = 1
        c = -1 * mean_returns  # we want to maximize the object function
        A = np.ones([1,len(mean_returns)]) # get coeff matrix [[1,1, ..,1]] 
        b = [1]  # the right-hand side
        return optimize.linprog(c, A_eq=A, b_eq=b, bounds=(0,1))

    def _minimize_risk(self, cov_returns: pd.DataFrame, no_of_buy: int):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        def f(x, cov_returns):
            # this is just the quadratic form x^T*Ax(but the direction is different to the math textbook...)
            return x @ cov_returns @ x.T
        
        def constraint_eq(x):
            # this is just [1, 1, ..., 1] * vector of transpose x(column vector) - 1 = 0 
            # because we want the sum of coeff to be 1
            return np.ones(x.shape) @ x.T - 1
        
        x0 = np.repeat(1/no_of_buy, no_of_buy)         # just take equal weights as the initial values 
        cons = ({'type': 'eq', 'fun': constraint_eq})  # the constraint in the above Kaggle tutorial
        bound = (0, 1)  # the interval of each weight
        bounds = tuple([bound for i in range(no_of_buy)])  # the parameter requires tuple type

        return optimize.minimize(f, x0=x0, args=(cov_returns), bounds=bounds, constraints=cons, tol=1.0e-3)

    # return the weights generated from mean-variance optimization
    def mean_variance_optimization(self, one_day_pred_top5: pd.DataFrame):
        df = self.train_model.df_full
        top5_returns = {}   # for the dataframe of 5 days future growth of top 5 stockes 
        
        for index, row in one_day_pred_top5.iterrows():
            returns = [] # for the list of 5 days returns of the corresponding ticker 
            for day in range(1, 6):
                filter = (df.Ticker == row.Ticker) & (df.Date == row.Date)  # the correspondin ticker and date
                returns.append(float(df[filter][f"growth_future_{day}d"]-1)) 
            top5_returns[row.Ticker] = returns

        top5_returns = pd.DataFrame(top5_returns)
        top5_returns.replace([np.inf, -np.inf], np.nan, inplace=True)  # if divided by 0 happens, view it as Nan
        top5_returns.fillna(0)   # if Nan, we just fill 0 to represent on return

        mean_returns = top5_returns.mean()
        cov_returns = top5_returns.cov()
        no_of_buy = len(one_day_pred_top5)  # total investments

        opt1 = self._maximize_returns(mean_returns)        # optimal parameters of maximize returns
        opt2 = self._minimize_risk(cov_returns, no_of_buy) # optimal parameters of minimize risk

        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize.html
        def f(x):
            # this is just the quadratic form x^T*Ax(but the direction is different to the math textbook...)
            return x @ cov_returns @ x.T
        
        def constraint_eq(x):
            # this is just [1, 1, ..., 1] * vector of transpose x(column vector) - 1 = 0 
            # because we want the sum of coeff to be 1
            return np.ones(x.shape) @ x.T - 1
        
        def constraint_ineq(x, R):
            # weighed average of mean of returns <= R
            return mean_returns @ x.T - R
        
        x0 = np.repeat(1/no_of_buy, no_of_buy)  # just take equal weights as the initial values
        R = (opt1.fun + opt2.fun) / 2           # take average of the two optimized returns
        # See the above Kaggle tutorial 
        cons = ({'type': 'eq', 'fun': constraint_eq}, 
                {'type': 'ineq', 'fun': constraint_ineq, 'args': (R,)})
        bound = (0, 1) # the interval of each weight
        bounds = tuple([bound for i in range(no_of_buy)]) ## the parameter requires tuple type

        opt = optimize.minimize(f, method='trust-constr', x0=x0, bounds=bounds, constraints=cons, tol=1.0e-3)
        return opt.x  # the optimal solution of weights is stored in x 
        

    # return the weights generated from Sharpe ratio optimization
    def sharpe_ratio_optimization(self, one_day_pred_top5: pd.DataFrame):
        pass