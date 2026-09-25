import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

print("==================================================")
print(" BLOCK 1: DATA INGESTION & ROBUST INVERSION ENGINE")
print("==================================================\n")

# ---------------------------------------------------------
# 1. THE PHYSICS: BLACK-SCHOLES MATH
# ---------------------------------------------------------
def bs_call_price(S, K, T, r, sigma):
    """Calculates the Black-Scholes price of a Call option."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_vega(S, K, T, r, sigma):
    """Calculates Vega (Sensitivity to volatility)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# ---------------------------------------------------------
# 2. THE ENGINE: ROBUST IV INVERSION (Resolves Reviewer Point 2)
# ---------------------------------------------------------
def implied_vol_robust(target_price, S, K, T, r, tol=1e-5, max_iter=100):
    """
    Safely converts a theoretical price back into Implied Volatility.
    Uses Newton-Raphson for speed, but falls back to Brent's method 
    if Vega is near zero (crucial for NIFTY weekly expiries).
    """
    # Safety Check: Is it priced below intrinsic value?
    intrinsic = max(0, S - K * np.exp(-r * T))
    if target_price <= intrinsic:
        return 1e-4 # Practically zero IV
        
    sigma = 0.20 # Start guessing at 20% IV
    
    # ATTEMPT 1: Newton-Raphson Method
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = price - target_price
        
        if abs(diff) < tol:
            return sigma
            
        # THE DANGER ZONE: Vega near zero (NIFTY weeklies)
        # If we divide by this, the math crashes. We must abort and use the fallback.
        if vega < 1e-6:
            break 
            
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4 # Keep strictly positive
            
    # ATTEMPT 2: The Bracketing Solver Fallback (Brent's Method)
    def objective(sig):
        return bs_call_price(S, K, T, r, sig) - target_price
        
    try:
        # Search for IV safely between 0.01% and 500%
        return brentq(objective, 1e-4, 5.0) 
    except ValueError:
        return np.nan # Option is unpriceable, flag for removal

# ---------------------------------------------------------
# 3. THE RADAR: PROCESSING A FAKE NSE OPTIONS CHAIN
# ---------------------------------------------------------
print("[*] Generating sample NSE NIFTY Options Data...")
# Simulating 5 option strikes. 
# Spot Price = 24000, 7 Days to Expiry (T=7/365), Risk-Free Rate = 5%
data = {
    'Strike': [23800, 23900, 24000, 24100, 24200],
    'Market_IV': [0.18, 0.16, 0.15, 0.145, 0.16],       # The actual market IV
    'Carr_Madan_Price': [280.5, 195.2, 125.0, 70.5, 35.2], # The 'Perfect' Price we calculated
    'Bid_Ask_Spread_IV': [0.01, 0.01, 0.005, 0.005, 0.02]  # Liquidity filter
}
df = pd.DataFrame(data)

S, T, r = 24000, 7/365, 0.05

print("[*] Inverting Carr-Madan Prices to Model IV...")
# Invert the prices cell by cell (Fix 1.1)
df['Model_IV'] = df.apply(lambda row: implied_vol_robust(row['Carr_Madan_Price'], S, row['Strike'], T, r), axis=1)

print("[*] Calculating Residuals (The Glitches)...")
# Residual = Market - Model (Fix 1.2)
df['Residual'] = df['Market_IV'] - df['Model_IV']

print("[*] Applying Liquidity Filter...")
# Only keep anomalies larger than half the bid-ask spread (Fix 1.7)
df['Is_Valid_Anomaly'] = abs(df['Residual']) > (0.5 * df['Bid_Ask_Spread_IV'])

print("\n--- FINAL RADAR OUTPUT ---")
print(df[['Strike', 'Market_IV', 'Model_IV', 'Residual', 'Is_Valid_Anomaly']])
print("\n[SUCCESS] Pipeline Phase 0 executed without crashing on NIFTY math.")
