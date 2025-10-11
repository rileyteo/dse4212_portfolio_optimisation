from imports import *
load_dotenv()

df = pd.read_csv('all_stocks_5yr.csv', index_col=0, parse_dates=True)
df = df.pivot_table(index='date', columns='Name', values='close').dropna(axis=1)
df = np.log(df).diff().dropna()

fred = Fred(api_key = os.getenv("API_KEY"))
risk_free_rate = fred.get_series_latest_release('DGS3MO')/100/252
risk_free_rate = risk_free_rate.dropna()
risk_free_rate = risk_free_rate.loc[df.index.min():df.index.max()]

df['risk_free_rate'] = risk_free_rate
train = df.iloc[:1000]
test = df.iloc[1000:].dropna()

df['risk_free_rate'].to_csv('risk_free_rate.csv')
df.iloc[:,:-1].to_csv('stock_returns.csv')
