from __init__ import *
from Utils import timer, timeformat, get_bucket_points
from Bucket import Bucket

class DataLoader(object):
    
    def __init__(self, date, symbol, dir_path, epsilon = 1e-6):

        self._date = date
        self._symbol = symbol
        self._dir_path = dir_path
        self.epsilon = epsilon

        self.load_data()
        self.data_expansion()
        self.data_preparation()
    
    @timer('Load Data')
    def load_data(self):

        file_path = self._dir_path + '\\'+ self._symbol + '.h5'
        hdf = pd.HDFStore(file_path, 'r')
        self.df = hdf.get(self._date)
        self.df['time'] = self.df['time'].apply(timeformat)
        self.df = self.df[['time', 'a1_v', 'a1_p', 'b1_v', 'b1_p', 'a2_v', 'a2_p', 'b2_v', 'b2_p']]
        self.df = self.df.set_index('time', drop=False) 
        self.df = self.df.drop_duplicates()

    @timer('Data Expansion')
    def data_expansion(self):
    
        spot_time = pd.to_datetime(self.df.index[0])
        end_time = pd.to_datetime(self._date + ' 14:50:00')
        time_interval = datetime.timedelta(seconds=1)

        time_range = []
        while spot_time < end_time:
            time_range.append(spot_time)
            spot_time += time_interval

        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        self.df = self.df.drop_duplicates(subset=['time'], keep='last') # keey latest data
        self.df = self.df.reindex(pd.Series(time_range))
        self.df = self.df.sort_index()
        self.df = self.df.ffill(axis=0)

    @timer('Features Computation 0')
    def data_preparation(self):

        self.df['WAP1'] = (self.df['b1_p']*self.df['a1_v'] + self.df['a1_p']*self.df['b1_v'])/(self.df['b1_v']+self.df['a1_v']).fillna(0) + self.epsilon
        self.df['WAP2'] = (self.df['b2_p']*self.df['a2_v'] + self.df['a2_p']*self.df['b2_v'])/(self.df['b2_v']+self.df['a2_v']).fillna(0) + self.epsilon
        self.df['r1'] = np.log(self.df['WAP1']/self.df['WAP1'].shift(1)).fillna(0)
        self.df['r2'] = np.log(self.df['WAP2']/self.df['WAP2'].shift(1)).fillna(0)

    def get_df(self):
        return self.df
        

class DataSet(object):
    
    def __init__(self, Stock_Info, dir_path, Daily_Start = ' 09:30:00', Daily_End = ' 14:59:59'):
        """
        Stock_Info: List(Tuple(str, List)) => [(stock id, [trading dates]), ...]
        """
        self.stock_info = Stock_Info
        self._dir_path = dir_path
        self.daily_start = Daily_Start
        self.daily_end = Daily_End
        self.WIN1 = 60
        self.WIN2 = 600
        self.n_cores = 1

    # @timer('Load Data==')
    def load_data(self, date, symbol):

        loader = DataLoader(date, symbol, self._dir_path) 
        self.df = loader.get_df()
        self.df = self.df[date + self.daily_start: date + self.daily_end]

        with timer('Rolling...'):
            self.df['WAP1_ROLLING_WIN2'] = list(self.df['WAP1'].rolling(self.WIN2))
            self.df['r1_ROLLING_WIN1']   = list(self.df['r1'].rolling(self.WIN1))
            self.df['r1_ROLLING_WIN2']   = list(self.df['r1'].rolling(self.WIN2))
            self.df['r2_ROLLING_WIN1']   = list(self.df['r2'].rolling(self.WIN1))
            self.df['r2_ROLLING_WIN2']   = list(self.df['r2'].rolling(self.WIN2))

        self.df['target_r']   = self.df['r1'].rolling(self.WIN2).sum().shift(-self.WIN2)
        self.df['target_vol'] = self.df['WAP1'].rolling(self.WIN2).std().shift(-self.WIN2)

    @timer('Features Computation 1: price impact')
    def get_price_impact(self):

        Ask_Price = (self.df['a1_p']*self.df['a1_v'] + self.df['a2_p']*self.df['a2_v']) / (self.df['a1_v'] + self.df['a2_v'])
        Bid_Price = (self.df['b1_p']*self.df['b1_v'] + self.df['b2_p']*self.df['b2_v']) / (self.df['b1_v'] + self.df['b2_v'])

        self.df['F1_1']  = (self.df['a1_p'] - Ask_Price) / self.df['a1_p']
        self.df['F1_2']  = (self.df['a2_p'] - Ask_Price) / self.df['a2_p']

        self.df['F1_3']  = (self.df['b1_p'] - Bid_Price) / self.df['b1_p']
        self.df['F1_4']  = (self.df['b2_p'] - Bid_Price) / self.df['b2_p']

    @timer('Features Computation 2: spread & imbalance')
    def get_spread(self):

        self.df['F2_1']  = self.df['b1_v'] - self.df['b2_v']
        self.df['F2_2']  = self.df['a1_v'] - self.df['a2_v']

        self.df['F2_3']  = (self.df['a1_p']  - self.df['b1_p'])/self.df['WAP1']
        self.df['F2_4']  = self.df['a1_v']  - self.df['b1_v']

        self.df['F2_5']  = (self.df['a2_p']  - self.df['b2_p'])/self.df['WAP2']
        self.df['F2_6']  = self.df['a2_v']  - self.df['b2_v']

        self.df['F2_7']  = np.log((self.df['a1_v']+self.df['a2_v']) / (self.df['b1_v']+self.df['b2_v']))

    @timer('Features Computation 3: depth')
    def get_depth(self):

        self.df['F3_1']  = self.df['a1_v'] + self.df['a2_v']
        self.df['F3_2']  = self.df['b1_v'] + self.df['b2_v']

    @timer('Features Computation 4: volatility')
    def get_volatility(self):

        def cal_vol(slice):
            return np.sqrt(np.power(slice, 2).sum())

        self.df['F4_1'] = self.df['r1_ROLLING_WIN1'].apply(lambda x: cal_vol(x))
        self.df['F4_2'] = self.df['r1_ROLLING_WIN2'].apply(lambda x: cal_vol(x))

    @timer('Features Computation 5: higher order moment')
    def get_high_order_moment(self):

        def cal_WAP_skew(slice, win_size):
            return (np.power((slice - slice.mean())/slice.std(), 3).sum()) / win_size

        self.df['F5_1']  = self.df['WAP1_ROLLING_WIN2'].apply(lambda x: cal_WAP_skew(x, self.WIN2))

    @timer('Features Computation 6: trend')
    def get_trend(self):
        
        self.df['F6_1'] = self.df['WAP1'].ewm(span=self.WIN1, adjust=False).mean() - self.df['WAP1'].ewm(span=self.WIN2, adjust=False).mean()

    @timer('Features Computation 7: momentum')
    def get_momentum(self):

        def cal_tsmom(slice, win_size):
            return max(np.cumsum(slice), key=abs) / win_size

        self.df['F7_1'] = self.df['r1_ROLLING_WIN1'].apply(lambda x: cal_tsmom(x, self.WIN1))
        self.df['F7_2'] = self.df['r1_ROLLING_WIN2'].apply(lambda x: cal_tsmom(x, self.WIN2))
      
    @timer('Features Computation')
    def get_all_basic_features(self):

        self.dataset = pd.DataFrame()
    
        for info in self.stock_info:
            stock_id = info[0]
            trading_dates = info[1]
            print(f'[!] Stock ID: {stock_id}')

            for date in trading_dates:
                print(f'[!] Trading date: {date}')
                self.load_data(date, stock_id) 
                self.get_price_impact() # 1
                self.get_spread() # 2
                self.get_depth() # 3
                self.get_volatility() # 4
                self.get_high_order_moment() # 5
                self.get_trend() # 6
                self.get_momentum() # 7
                self.df = self.df.sort_index()
                self.df['time'] = self.df.index
                self.df = self.df[date+' 09:40:00':date+' 14:40:00']

                self.dataset = self.dataset.append(self.df)
                self.df = pd.DataFrame()
            
    def dataset_shuffle(self, _df_):
        return shuffle(_df_)
         
    def sample(self, _df_, _frac):
        return _df_.sample(frac=_frac)

    def get_items(self):
        return self.dataset
    