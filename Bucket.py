from sympy import symarray
from __init__ import  *
from Utils import *


class Bucket(object):
    def __init__(self, start_time, header, data, trailer):

        self.start_time = start_time
        self.header = header
        self.df = data
        self.joint_bucket = self.header.append(self.df)

        self.trailer = trailer
        self.df['target'] = np.sqrt(self.trailer['r1'].apply(lambda x: x**2).sum())
        self.df['baseline'] = np.sqrt(self.df['r1'].apply(lambda x: x**2).sum())
    
    # @timer('Features Computation 1')
    def get_price_impact(self):

        Ask_Price = (self.df['a1_p']*self.df['a1_v'] + self.df['a2_p']*self.df['a2_v']) / (self.df['a1_v'] + self.df['a2_v'])
        Bid_Price = (self.df['b1_p']*self.df['b1_v'] + self.df['b2_p']*self.df['b2_v']) / (self.df['b1_v'] + self.df['b2_v'])

        self.df['F1_1']  = (self.df['a1_p'] - Ask_Price) / self.df['a1_p']
        self.df['F1_2']  = (self.df['a2_p'] - Ask_Price) / self.df['a2_p']

        self.df['F1_3']  = (self.df['b1_p'] - Bid_Price) / self.df['b1_p']
        self.df['F1_4']  = (self.df['b2_p'] - Bid_Price) / self.df['b2_p']

    # @timer('Features Computation 2')
    def get_spread(self):

        self.df['F2_1']  = self.df['b1_p'] - self.df['b2_p']
        self.df['F2_2']  = np.abs(self.df['b1_v'] - self.df['b2_v'])
        self.df['F2_3']  = np.abs(self.df['b1_p']*self.df['b1_v'] - self.df['b2_p']*self.df['b2_v'])

        self.df['F2_4']  = self.df['a1_p'] - self.df['a2_p']
        self.df['F2_5']  = np.abs(self.df['a1_v'] - self.df['a2_v'])
        self.df['F2_6']  = np.abs(self.df['a1_p']*self.df['a1_v'] - self.df['a2_p']*self.df['a2_v'])

        self.df['F2_7']  = self.df['F2_1'] + self.df['F2_4']
        self.df['F2_8']  = self.df['F2_2'] + self.df['F2_5']
        self.df['F2_9']  = self.df['F2_3'] + self.df['F2_6']

        self.df['F2_10'] = self.df['a1_p']  - self.df['b1_p']
        self.df['F2_11'] = np.abs(self.df['a1_v']  - self.df['b1_v'])
        self.df['F2_12'] = np.abs(self.df['a1_p']*self.df['a1_v'] - self.df['b1_p']*self.df['b1_v'])

        self.df['F2_13'] = self.df['a2_p']  - self.df['b2_p']
        self.df['F2_14'] = np.abs(self.df['a2_v']  - self.df['b2_v'])
        self.df['F2_15'] = np.abs(self.df['a2_p']*self.df['a2_v'] - self.df['b2_p']*self.df['b2_v'])

        self.df['F2_16'] = np.abs(self.df['F1_1'] - self.df['F1_3'])
        self.df['F2_17'] = np.abs(self.df['F1_2'] - self.df['F1_4'])

    # @timer('Features Computation 3')
    def get_depth(self):

        self.df['F3_1']  = self.df['a1_v'] + self.df['a2_v']
        self.df['F3_2']  = self.df['a1_p']*self.df['a1_v'] +  self.df['a2_p']*self.df['a2_v']
 
        self.df['F3_3']  = self.df['b1_v'] + self.df['b2_v']
        self.df['F3_4']  = self.df['b1_p']*self.df['b1_v'] +  self.df['b2_p']*self.df['b2_v']    
 
        self.df['F3_5']  = self.df['F3_1'] + self.df['F3_3']
        self.df['F3_6']  = self.df['F3_2'] + self.df['F3_4']

    # @timer('Features Computation 4')
    def get_imbalance(self):

        self.df['F4_1']  = np.abs((self.df['a1_v']+self.df['a2_v']) - (self.df['b1_v']+self.df['b2_v']))
        self.df['F4_2']  = np.abs((self.df['a1_p']*self.df['a1_v'] + self.df['a2_p']*self.df['a2_v']) - (self.df['b1_p']*self.df['b1_v'] + self.df['b2_p']*self.df['b2_v']))
 
        self.df['F4_3']  = (self.df['a1_v']+self.df['a2_v']) / (self.df['b1_v']+self.df['b2_v'])
        self.df['F4_4']  = (self.df['a1_p']*self.df['a1_v'] + self.df['a2_p']*self.df['a2_v']) / (self.df['b1_p']*self.df['b1_v'] + self.df['b2_p']*self.df['b2_v'])

    # @timer('Features Computation 5')
    def get_volatility(self):

        self.df['F5_1']  = np.sqrt(self.df['r1'].apply(lambda x: x**2).sum())
        self.df['F5_2']  = np.sqrt(self.joint_bucket['r1'].apply(lambda x: x**2).sum())
 
        self.df['F5_3']  = np.sqrt(self.df['r2'].apply(lambda x: x**2).sum())
        self.df['F5_4']  = np.sqrt(self.joint_bucket['r2'].apply(lambda x: x**2).sum())

    # @timer('Features Computation 6')
    def get_high_order_moment(self):

        self.df['F6_1']  = ((self.df['WAP1'] - self.df['WAP1'].mean())/self.df['WAP1'].std()).apply(lambda x: x**3).sum() / self.df.shape[0]
        self.df['F6_2']  = ((self.joint_bucket['WAP1'] - self.joint_bucket['WAP1'].mean())/self.joint_bucket['WAP1'].std()).apply(lambda x: x**3).sum() / self.joint_bucket.shape[0]
        self.df['F6_3']  = ((self.df['WAP2'] - self.df['WAP2'].mean())/self.df['WAP2'].std()).apply(lambda x: x**3).sum() / self.df.shape[0]
        self.df['F6_4']  = ((self.joint_bucket['WAP2'] - self.joint_bucket['WAP2'].mean())/self.joint_bucket['WAP2'].std()).apply(lambda x: x**3).sum() / self.joint_bucket.shape[0]

        self.df['F6_5']  = np.power(np.abs(self.df['r1'].apply(lambda x: x**3).sum()), 1/3)
        self.df['F6_6']  = np.power(np.abs(self.joint_bucket['r1'].apply(lambda x: x**3).sum()), 1/3)
        self.df['F6_7']  = np.power(np.abs(self.df['r2'].apply(lambda x: x**3).sum()), 1/3)
        self.df['F6_8']  = np.power(np.abs(self.joint_bucket['r2'].apply(lambda x: x**3).sum()), 1/3)

        self.df['F6_9']  = ((self.df['WAP1'] - self.df['WAP1'].mean())/self.df['WAP1'].std()).apply(lambda x: x**4).sum() / self.df.shape[0]
        self.df['F6_10'] = ((self.joint_bucket['WAP1'] - self.joint_bucket['WAP1'].mean())/self.joint_bucket['WAP1'].std()).apply(lambda x: x**4).sum() / self.joint_bucket.shape[0]
        self.df['F6_11'] = ((self.df['WAP2'] - self.df['WAP2'].mean())/self.df['WAP2'].std()).apply(lambda x: x**4).sum() / self.df.shape[0]
        self.df['F6_12'] = ((self.joint_bucket['WAP2'] - self.joint_bucket['WAP2'].mean())/self.joint_bucket['WAP2'].std()).apply(lambda x: x**4).sum() / self.joint_bucket.shape[0]

        self.df['F6_13'] = np.power(self.df['r1'].apply(lambda x: x**4).sum(), 1/4)
        self.df['F6_14'] = np.power(self.joint_bucket['r1'].apply(lambda x: x**4).sum(), 1/4)
        self.df['F6_15'] = np.power(self.df['r2'].apply(lambda x: x**4).sum(), 1/4)
        self.df['F6_16'] = np.power(self.joint_bucket['r2'].apply(lambda x: x**4).sum(), 1/4)

    # @timer('Features Computation 7')
    def get_momentum(self):

        self.df['F7_1']  = max(np.cumprod((self.df['r1']+1e-6)/self.df['r1'].std()), key=abs)
        self.df['F7_2']  = max(np.cumprod((self.joint_bucket['r1']+1e-6)/self.joint_bucket['r1'].std()), key=abs)
        self.df['F7_3']  = max(np.cumprod((self.df['r2']+1e-6)/self.df['r2'].std()), key=abs)
        self.df['F7_4']  = max(np.cumprod((self.joint_bucket['r2']+1e-6)/self.joint_bucket['r2'].std()), key=abs)  

        self.df['F7_5']  = max(np.cumsum(np.sign(self.df['r1'])), key=abs)/self.df.shape[0]
        self.df['F7_6']  = max(np.cumsum(np.sign(self.joint_bucket['r1'])), key=abs)/self.joint_bucket.shape[0]
        self.df['F7_7']  = max(np.cumsum(np.sign(self.df['r2'])), key=abs)/self.df.shape[0]
        self.df['F7_8']  = max(np.cumsum(np.sign(self.joint_bucket['r2'])), key=abs)/self.joint_bucket.shape[0]  

    # @timer('Features Computation 1~7 total')
    def get_all_basic_features(self):

        self.get_price_impact()
        self.get_spread()
        self.get_depth()
        self.get_imbalance()
        self.get_volatility()
        self.get_high_order_moment()
        self.get_momentum()
        

    def get_df(self):
        return self.df

    def plot_corr(self, features):
        
        self.Features = self.df[features]
        plot_features_corr(self.Features)

    def plot_order_book(self):
        
        n_rows = 2
        _, ax = plt.subplots(n_rows, figsize=(21, 14), dpi=300)
        ax[0].set_title(f"{self.df['time'][0]} -- {self.df['time'][-1]}", fontsize=16)
        ax[0].plot(self.df['a1_p'], color='r', label = 'Ask1 price', linewidth = 0.5)
        ax[0].plot(self.df['a2_p'], color='orange', label = 'Ask2 price', linewidth = 0.5)
        ax[0].plot(self.df['b1_p'], color='g', label = 'Bid1 price', linewidth = 0.5)
        ax[0].plot(self.df['b2_p'], color='blue', label = 'Bid2 price', linewidth = 0.5)
        ax[0].plot(self.df['WAP1'], color='pink', label = 'WAP1', linewidth = 0.5, linestyle='--')
        ax[0].plot(self.df['WAP2'], color='purple', label = 'WAP2', linewidth = 0.5, linestyle='--', alpha = 0.8)
        ax[0].get_xaxis().set_visible(False)
        ax[0].legend()
        ax[1].stackplot([i for i in range(600)], self.df['a1_v'], self.df['a2_v'], self.df['b1_v'], self.df['b2_v'], colors = ['r', 'orange', 'g', 'b'], alpha = 0.6, labels = ['Ask1 Volume', 'Ask2 Volume', 'Bid1 Volume', 'Bid2 Volume'])
        ax[1].legend()
        plt.tight_layout()
