from imports import *
load_dotenv()

df = pd.read_csv('all_stocks_5yr.csv', index_col=0, parse_dates=True)

close = df.pivot_table(index='date', columns='Name', values='close').dropna(axis=1)
op = df.pivot_table(index='date', columns='Name', values='open')[[col for col in close.columns]]
high = df.pivot_table(index='date', columns='Name', values='high')[[col for col in close.columns]]
low = df.pivot_table(index='date', columns='Name', values='low')[[col for col in close.columns]]
volume = df.pivot_table(index='date', columns='Name', values='volume')[[col for col in close.columns]]

op.fillna(method='ffill', inplace=True)
high.fillna(method='ffill', inplace=True)
low.fillna(method='ffill', inplace=True)
volume.replace(0, np.nan, inplace=True)
volume.fillna(method='ffill', inplace=True)

returns = np.log(close).diff().dropna()

fred = Fred(api_key = os.getenv("API_KEY"))
risk_free_rate = fred.get_series_latest_release('DGS3MO')/100/252
risk_free_rate = risk_free_rate.reindex(returns.index, method='ffill')


with open('processed_data.pkl', 'wb') as f:
    pickle.dump((op, high, low, close, volume, returns, risk_free_rate), f)
