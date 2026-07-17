import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.stats as stats
import statsmodels.api as sm

# Define Tickers
TICKERS = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "AVGO", "META", "TSLA", "LLY", "MU"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def ensure_data_dir():
    """Create data caching directory if it doesn't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def get_historical_prices(symbol, start_date="2016-01-01"):
    """Fetch historical daily stock prices for a symbol."""
    print(f"Fetching historical prices for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=datetime.now().strftime("%Y-%m-%d"))
    # Clean index
    df.index = df.index.tz_localize(None).normalize()
    return df[['Close']]

def get_analyst_targets(symbol, start_date="2016-01-01"):
    """Fetch and clean historical analyst upgrades, downgrades, and price targets."""
    print(f"Fetching analyst targets for {symbol}...")
    ticker = yf.Ticker(symbol)
    ud = ticker.upgrades_downgrades
    if ud is None or ud.empty:
        print(f"No analyst targets found for {symbol}")
        return pd.DataFrame()
    
    # Process index and reset
    df = ud.reset_index()
    df['GradeDate'] = pd.to_datetime(df['GradeDate']).dt.tz_localize(None).dt.normalize()
    
    # Filter columns and clean targets
    required_cols = ['GradeDate', 'Firm', 'currentPriceTarget', 'priorPriceTarget']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[['GradeDate', 'Firm', 'currentPriceTarget', 'priorPriceTarget']].dropna(subset=['currentPriceTarget'])
    df = df[df['currentPriceTarget'] > 0]
    
    # Filter by start date
    df = df[df['GradeDate'] >= pd.to_datetime(start_date)]
    df = df.sort_values('GradeDate').reset_index(drop=True)
    return df

def align_targets_and_prices(targets_df, prices_df):
    """Align price targets with stock prices at issuance and 12 months later."""
    if targets_df.empty or prices_df.empty:
        return pd.DataFrame()
    
    aligned_rows = []
    
    # Sort prices for fast lookup
    prices_df = prices_df.sort_index()
    all_dates = prices_df.index
    
    for _, row in targets_df.iterrows():
        target_date = row['GradeDate']
        target_price = row['currentPriceTarget']
        firm = row['Firm']
        
        # 1. Price at target issuance
        # Find closest trading day on or after target_date
        idx = all_dates.searchsorted(target_date)
        if idx >= len(all_dates):
            # Target was set after our price series ended
            continue
        price_date = all_dates[idx]
        price_at_target = prices_df.loc[price_date, 'Close']
        
        # 2. Forward price 12 months later
        target_forward_date = target_date + timedelta(days=365)
        idx_forward = all_dates.searchsorted(target_forward_date)
        if idx_forward >= len(all_dates):
            # 12 months forward is in the future relative to our data (incomplete period)
            price_12m_forward = np.nan
            forward_date = pd.NaT
        else:
            forward_date = all_dates[idx_forward]
            price_12m_forward = prices_df.loc[forward_date, 'Close']
            
        aligned_rows.append({
            'ticker_date': target_date,
            'Firm': firm,
            'TargetPrice': target_price,
            'StockPriceAtTarget': price_at_target,
            'ForwardDate': forward_date,
            'StockPrice12mForward': price_12m_forward
        })
        
    return pd.DataFrame(aligned_rows)

def compute_rolling_consensus(symbol, targets_df, prices_df, window_days=180):
    """
    Compute daily rolling consensus price targets.
    Consensus is defined as the average of targets issued in the last 180 days.
    """
    if targets_df.empty or prices_df.empty:
        return pd.DataFrame()
    
    # Create daily date range from min target to latest price
    start_date = targets_df['GradeDate'].min()
    end_date = prices_df.index.max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Set targets index
    targets_df = targets_df.set_index('GradeDate')
    
    consensus_history = []
    
    # Iterate daily and compute consensus
    for current_date in date_range:
        cutoff_date = current_date - timedelta(days=window_days)
        # Fetch targets within the window
        active_targets = targets_df.loc[cutoff_date:current_date]
        
        if active_targets.empty:
            consensus_target = np.nan
        else:
            # Take average of currentPriceTarget for active targets
            consensus_target = active_targets['currentPriceTarget'].mean()
            
        consensus_history.append({
            'Date': current_date,
            'ConsensusTarget': consensus_target
        })
        
    consensus_df = pd.DataFrame(consensus_history).set_index('Date')
    
    # Align with actual stock prices
    aligned = consensus_df.join(prices_df, how='inner')
    
    # Match with 12m forward actual prices
    aligned_rows = []
    all_dates = aligned.index
    
    for date, row in aligned.iterrows():
        consensus = row['ConsensusTarget']
        stock_price = row['Close']
        
        if pd.isnull(consensus):
            continue
            
        target_forward_date = date + timedelta(days=365)
        idx_forward = all_dates.searchsorted(target_forward_date)
        if idx_forward >= len(all_dates):
            price_12m_forward = np.nan
            forward_date = pd.NaT
        else:
            forward_date = all_dates[idx_forward]
            price_12m_forward = aligned.loc[forward_date, 'Close']
            
        aligned_rows.append({
            'Date': date,
            'ConsensusTarget': consensus,
            'StockPrice': stock_price,
            'ForwardDate': forward_date,
            'StockPrice12mForward': price_12m_forward
        })
        
    return pd.DataFrame(aligned_rows)

def run_extraction_pipeline():
    """Extract and process data for all tickers and cache them in CSV files."""
    ensure_data_dir()
    print("Starting PARA Data Extraction Pipeline...")
    
    for symbol in TICKERS:
        print(f"\nProcessing {symbol}...")
        try:
            # 1. Fetch Prices & Targets
            prices_df = get_historical_prices(symbol)
            targets_df = get_analyst_targets(symbol)
            
            if targets_df.empty:
                print(f"Skipping {symbol} due to missing target data.")
                continue
                
            # Cache raw data
            prices_df.to_csv(os.path.join(DATA_DIR, f"{symbol}_prices.csv"))
            targets_df.to_csv(os.path.join(DATA_DIR, f"{symbol}_targets_raw.csv"), index=False)
            
            # 2. Align individual targets
            individual_aligned = align_targets_and_prices(targets_df, prices_df)
            individual_aligned.to_csv(os.path.join(DATA_DIR, f"{symbol}_individual_aligned.csv"), index=False)
            print(f"  Saved {len(individual_aligned)} individual aligned recommendations.")
            
            # 3. Compute rolling consensus
            consensus_aligned = compute_rolling_consensus(symbol, targets_df, prices_df)
            consensus_aligned.to_csv(os.path.join(DATA_DIR, f"{symbol}_consensus_aligned.csv"), index=False)
            print(f"  Saved rolling consensus data.")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            
    print("\nExtraction and caching completed successfully!")

def load_processed_data(symbol, data_type='individual'):
    """Utility to load processed aligned data for a ticker."""
    filename = f"{symbol}_{data_type}_aligned.csv"
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Parse dates
        for col in ['Date', 'ticker_date', 'ForwardDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    else:
        print(f"File not found: {path}")
        return pd.DataFrame()

def calculate_statistical_metrics(df):
    """
    Calculate statistical metrics of accuracy, bias, and calibration.
    Input df can be individual recommendations or consensus-sampled.
    """
    # Clean rows with missing actual price forward (targets set in the last 12m)
    df_clean = df.dropna(subset=['StockPrice12mForward']).copy()
    if df_clean.empty:
        return {}
    
    # 1. Column normalization
    if 'TargetPrice' in df_clean.columns:
        target_col = 'TargetPrice'
        price_at_target_col = 'StockPriceAtTarget'
    else:
        target_col = 'ConsensusTarget'
        price_at_target_col = 'StockPrice'
        
    forward_price_col = 'StockPrice12mForward'
    
    # 2. Computations
    # Returns
    df_clean['target_implied_return'] = (df_clean[target_col] - df_clean[price_at_target_col]) / df_clean[price_at_target_col]
    df_clean['actual_12m_return'] = (df_clean[forward_price_col] - df_clean[price_at_target_col]) / df_clean[price_at_target_col]
    
    # Forecast errors (relative to actual price)
    # positive means target was too high (optimistic)
    df_clean['forecast_error'] = (df_clean[target_col] - df_clean[forward_price_col]) / df_clean[forward_price_col]
    df_clean['abs_forecast_error'] = df_clean['forecast_error'].abs()
    
    # Hit rate: was actual price >= target price at 12-month mark?
    df_clean['target_met'] = df_clean[forward_price_col] >= df_clean[target_col]
    
    # 3. Metrics
    mape = df_clean['abs_forecast_error'].mean() * 100
    mfe = df_clean['forecast_error'].mean() * 100
    hit_rate = df_clean['target_met'].mean() * 100
    rmse = np.sqrt((df_clean['forecast_error'] ** 2).mean()) * 100
    
    # Correlation between target returns and actual returns
    pearson_r, pearson_p = stats.pearsonr(df_clean['target_implied_return'], df_clean['actual_12m_return'])
    spearman_r, spearman_p = stats.spearmanr(df_clean['target_implied_return'], df_clean['actual_12m_return'])
    
    # OLS regression of actual return on target return
    X = sm.add_constant(df_clean['target_implied_return'])
    y = df_clean['actual_12m_return']
    model = sm.OLS(y, X).fit()
    
    beta = model.params.get('target_implied_return', np.nan)
    alpha = model.params.get('const', np.nan)
    p_value = model.pvalues.get('target_implied_return', np.nan)
    r_squared = model.rsquared
    
    return {
        'count': len(df_clean),
        'mape': mape,
        'mfe': mfe,
        'rmse': rmse,
        'hit_rate': hit_rate,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'ols_alpha': alpha,
        'ols_beta': beta,
        'ols_p_value': p_value,
        'r_squared': r_squared
    }

if __name__ == "__main__":
    import sys
    # If run directly, check if we should run the pipeline
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running dry-run test on AAPL...")
        ensure_data_dir()
        prices = get_historical_prices("AAPL")
        targets = get_analyst_targets("AAPL")
        aligned = align_targets_and_prices(targets, prices)
        print("Success! Sample aligned data:")
        print(aligned.head())
        metrics = calculate_statistical_metrics(aligned)
        print("Sample metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        run_extraction_pipeline()
