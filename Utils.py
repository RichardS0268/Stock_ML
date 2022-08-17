from __init__ import *

# timestamp to string
def timeformat(_time):

    _time = str(_time)
    return _time[:4] + '-' + _time[4:6] + '-' + _time[6:8] + ' ' + _time[8:10] + ':' + _time[10:12] + ':' + _time[12:14] 



@contextmanager 
def timer(name: str): # ⏱

    s = time.time()
    yield
    elapsed = time.time() - s
    print(f'[{name}] {elapsed: .3f} seconds')



def get_bucket_points(date):

    bucket_points = [pd.to_datetime(date + ' 09:30:00'), pd.to_datetime(date + ' 09:40:00'), pd.to_datetime(date + ' 09:50:00'), \
                     pd.to_datetime(date + ' 10:00:00'), pd.to_datetime(date + ' 10:10:00'), pd.to_datetime(date + ' 10:20:00'), \
                     pd.to_datetime(date + ' 10:30:00'), pd.to_datetime(date + ' 10:40:00'), pd.to_datetime(date + ' 10:50:00'), \
                     pd.to_datetime(date + ' 11:00:00'), pd.to_datetime(date + ' 11:10:00'), pd.to_datetime(date + ' 11:20:00'), \
                     pd.to_datetime(date + ' 13:00:00'), pd.to_datetime(date + ' 13:10:00'), pd.to_datetime(date + ' 13:20:00'), \
                     pd.to_datetime(date + ' 13:30:00'), pd.to_datetime(date + ' 13:40:00'), pd.to_datetime(date + ' 13:50:00'), \
                     pd.to_datetime(date + ' 14:00:00'), pd.to_datetime(date + ' 14:10:00'), pd.to_datetime(date + ' 14:20:00'), \
                     pd.to_datetime(date + ' 14:30:00'), pd.to_datetime(date + ' 14:40:00')]
                    
    return bucket_points



def plot_features_corr(features):

    f = plt.figure(figsize=(10, 7), dpi=130)
    plt.matshow(features.corr(), fignum=f.number)
    plt.xticks(range(features.select_dtypes(['number']).shape[1]), features.select_dtypes(['number']).columns, fontsize=5, rotation=60)
    plt.yticks(range(features.select_dtypes(['number']).shape[1]), features.select_dtypes(['number']).columns, fontsize=5)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=6)
    plt.title('Correlation Matrix', fontsize=8)



def plot_importance(cvbooster,  features, figsize=(10, 10), top_nums=50):
    raw_importances = cvbooster.feature_importance(importance_type='gain')
    # feature_name = cvbooster.boosters[0].feature_name()
    importance_df = pd.DataFrame(data=raw_importances,
                                 columns=features)
    # order by average importance across folds
    sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
    sorted_importance_df = importance_df.loc[:, sorted_indices]
    # plot top-n
    plot_cols = sorted_importance_df.columns[:top_nums]
    _, ax = plt.subplots(figsize=figsize)
    ax.grid()
    ax.set_xscale('log')
    ax.set_ylabel('Feature')
    ax.set_xlabel('Importance')
    sns.boxplot(data=sorted_importance_df[plot_cols],
                orient='h',
                ax=ax)
    plt.show()