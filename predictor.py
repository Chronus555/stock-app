import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost; fall back to sklearn GradientBoosting
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class StockPredictor:
    def __init__(self, ticker, period='2y', horizon=5):
        self.ticker = ticker
        self.period = period
        self.horizon = horizon
        self.df = None
        self.scaler = StandardScaler()
        self.rf_model = None
        self.xgb_model = None
        self.gb_model = None
        self.meta_model = None
        self.feature_cols = []
        self.selected_features = []
        self.df_clean = None

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        self.df = stock.history(period=self.period)
        if self.df.empty:
            raise ValueError(f"No data found for ticker '{self.ticker}'")
        self.df.index = pd.to_datetime(self.df.index)
        return self.df

    # ------------------------------------------------------------------
    # Classic indicator helpers
    # ------------------------------------------------------------------
    def _rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(self, df, period=14):
        hl  = df['High'] - df['Low']
        hpc = (df['High'] - df['Close'].shift(1)).abs()
        lpc = (df['Low']  - df['Close'].shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    def _obv(self, df):
        sign = np.sign(df['Close'].diff().fillna(0))
        return (sign * df['Volume']).cumsum()

    def _stochastic(self, df, k=14, d=3):
        low_min  = df['Low'].rolling(k).min()
        high_max = df['High'].rolling(k).max()
        denom    = (high_max - low_min).replace(0, np.nan)
        K        = 100 * (df['Close'] - low_min) / denom
        D        = K.rolling(d).mean()
        return K, D

    def _williams_r(self, df, period=14):
        high_max = df['High'].rolling(period).max()
        low_min  = df['Low'].rolling(period).min()
        denom    = (high_max - low_min).replace(0, np.nan)
        return -100 * (high_max - df['Close']) / denom

    def _cci(self, df, period=20):
        tp      = (df['High'] + df['Low'] + df['Close']) / 3
        mean_tp = tp.rolling(period).mean()
        mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (tp - mean_tp) / (0.015 * mean_dev.replace(0, np.nan))

    def _adx(self, df, period=14):
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        plus_dm_clean = np.where(plus_dm > minus_dm, plus_dm, 0)
        minus_dm_clean = np.where(minus_dm > plus_dm, minus_dm, 0)
        plus_dm_s = pd.Series(plus_dm_clean, index=df.index)
        minus_dm_s = pd.Series(minus_dm_clean, index=df.index)
        atr = self._atr(df, period)
        plus_di = 100 * plus_dm_s.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm_s.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx, plus_di, minus_di

    def _mfi(self, df, period=14):
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        mf = tp * df['Volume']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        mfr = pos_mf / neg_mf.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    def _roc(self, prices, period=12):
        return ((prices - prices.shift(period)) / prices.shift(period).replace(0, np.nan)) * 100

    # ------------------------------------------------------------------
    # NEW: Hurst Exponent (rescaled range method)
    # ------------------------------------------------------------------
    def _hurst_exponent(self, df, window=100):
        """
        Hurst Exponent — measures long-term memory of a time series.

        H > 0.5 → persistent/trending (momentum works)
        H = 0.5 → random walk (nothing works)
        H < 0.5 → anti-persistent/mean-reverting (contrarian works)

        This is the key indicator that Renaissance Technologies used to detect
        exploitable patterns: trade momentum when H>0.5, mean-reversion when H<0.5.
        """
        prices = df['Close'].values
        hurst = np.full(len(prices), np.nan)

        for i in range(window, len(prices)):
            seg = prices[i - window:i]
            log_returns = np.diff(np.log(seg))
            if len(log_returns) < 20:
                continue

            # Rescaled range for multiple sub-divisions
            max_k = min(len(log_returns) // 2, 50)
            if max_k < 8:
                hurst[i] = 0.5
                continue

            ns = []
            rs_values = []
            for n in range(8, max_k + 1):
                num_blocks = len(log_returns) // n
                if num_blocks < 1:
                    continue
                rs_list = []
                for b in range(num_blocks):
                    block = log_returns[b * n:(b + 1) * n]
                    mean_block = np.mean(block)
                    cumdev = np.cumsum(block - mean_block)
                    R = np.max(cumdev) - np.min(cumdev)
                    S = np.std(block, ddof=1)
                    if S > 0:
                        rs_list.append(R / S)
                if rs_list:
                    ns.append(n)
                    rs_values.append(np.mean(rs_list))

            if len(ns) >= 3:
                log_ns = np.log(ns)
                log_rs = np.log(rs_values)
                # Linear regression slope = Hurst exponent
                slope = np.polyfit(log_ns, log_rs, 1)[0]
                hurst[i] = np.clip(slope, 0.0, 1.0)
            else:
                hurst[i] = 0.5

        return pd.Series(hurst, index=df.index)

    # ------------------------------------------------------------------
    # NEW: Z-Score Mean Reversion Signal
    # ------------------------------------------------------------------
    def _zscore_mean_reversion(self, df, window=20):
        """
        Z-score of price relative to its rolling mean.
        Extreme negative z-scores → oversold → expect bounce (buy signal)
        Extreme positive z-scores → overbought → expect pullback (sell signal)
        """
        mean = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        return (df['Close'] - mean) / std.replace(0, np.nan)

    # ------------------------------------------------------------------
    # NEW: Return Autocorrelation
    # ------------------------------------------------------------------
    def _autocorrelation(self, df, lag=1, window=20):
        """
        Rolling autocorrelation of returns.
        Positive autocorrelation → momentum regime (trends persist)
        Negative autocorrelation → mean-reversion regime (reversals)
        Near zero → random walk
        """
        returns = df['Close'].pct_change()
        return returns.rolling(window).apply(
            lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag + 1 else 0,
            raw=False
        )

    # ------------------------------------------------------------------
    # NEW: Order Flow Imbalance Proxy
    # ------------------------------------------------------------------
    def _order_flow_imbalance(self, df, window=14):
        """
        Estimates buy/sell pressure using price direction + volume.
        Positive → buying pressure dominates
        Negative → selling pressure dominates

        This is a proxy for actual order flow (which requires tick data).
        Volume on up-moves is treated as buying; volume on down-moves as selling.
        """
        price_change = df['Close'].diff()
        # Classify volume as buying or selling based on close-to-close direction
        buy_vol = df['Volume'].where(price_change > 0, 0)
        sell_vol = df['Volume'].where(price_change < 0, 0)

        buy_pressure = buy_vol.rolling(window).sum()
        sell_pressure = sell_vol.rolling(window).sum()
        total = (buy_pressure + sell_pressure).replace(0, np.nan)

        # Normalized imbalance [-1, 1]
        ofi = (buy_pressure - sell_pressure) / total
        return ofi

    # ------------------------------------------------------------------
    # NEW: Multi-Timeframe Momentum Convergence
    # ------------------------------------------------------------------
    def _mtf_momentum_convergence(self, df):
        """
        When momentum across multiple timeframes aligns in the same direction,
        the signal is much stronger. This indicator measures convergence.

        Range: -1 (all bearish) to +1 (all bullish)
        Near 0 → conflicting signals across timeframes → low confidence
        """
        periods = [5, 10, 20, 50]
        signals = []
        for p in periods:
            mom = df['Close'] / df['Close'].shift(p) - 1
            signals.append(np.sign(mom))

        # Average of signs: +1 if all bullish, -1 if all bearish
        convergence = sum(signals) / len(signals)
        return convergence

    # ------------------------------------------------------------------
    # NEW: Volatility Regime Indicator
    # ------------------------------------------------------------------
    def _volatility_regime(self, df, window=20, lookback=252):
        """
        Percentile rank of current volatility vs historical.
        High percentile → volatile regime → larger moves, higher risk
        Low percentile → calm regime → smaller moves
        """
        vol = df['Close'].pct_change().rolling(window).std()
        return vol.rolling(lookback, min_periods=50).rank(pct=True)

    # ------------------------------------------------------------------
    # NEW: Relative Volume Spike Detector
    # ------------------------------------------------------------------
    def _volume_spike_zscore(self, df, window=20):
        """
        Z-score of current volume vs rolling average.
        Spikes in volume often precede or confirm significant moves.
        """
        vol_mean = df['Volume'].rolling(window).mean()
        vol_std = df['Volume'].rolling(window).std()
        return (df['Volume'] - vol_mean) / vol_std.replace(0, np.nan)

    # ------------------------------------------------------------------
    # NEW: Intraday Range Ratio
    # ------------------------------------------------------------------
    def _range_ratio(self, df, window=14):
        """
        Ratio of (Close-Open) to (High-Low).
        Near +1 → strong bullish candle (closed near high)
        Near -1 → strong bearish candle (closed near low)
        Near 0 → indecision/doji
        """
        body = df['Close'] - df['Open']
        shadow = (df['High'] - df['Low']).replace(0, np.nan)
        raw = body / shadow
        return raw.rolling(window).mean()

    # ------------------------------------------------------------------
    # AEMI (original novel indicator)
    # ------------------------------------------------------------------
    def _adaptive_entropy_momentum_index(self, df, entropy_window=20,
                                          fast=10, slow=30):
        log_ret = np.log(df['Close'] / df['Close'].shift(1))

        def _entropy(series):
            arr = series.dropna().values
            if len(arr) < 5:
                return np.nan
            n_bins = max(5, int(np.sqrt(len(arr))))
            counts, _ = np.histogram(arr, bins=n_bins)
            counts = counts[counts > 0]
            probs = counts / counts.sum()
            H = -np.sum(probs * np.log2(probs))
            H_max = np.log2(n_bins)
            return H / H_max if H_max > 0 else 1.0

        entropy = log_ret.rolling(entropy_window).apply(_entropy, raw=False)
        predictability = (1 - entropy).clip(0, 1)

        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        momentum = ema_fast / ema_slow.replace(0, np.nan) - 1

        vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
        vol_part  = np.tanh(vol_ratio.fillna(1) - 1)

        return momentum * predictability * (1 + 0.35 * vol_part)

    # ------------------------------------------------------------------
    # Fractal Dimension
    # ------------------------------------------------------------------
    def _fractal_dimension(self, df, window=20):
        prices = df['Close'].values
        fd = np.full(len(prices), np.nan)
        for i in range(window, len(prices)):
            seg = prices[i - window: i]
            price_range = seg.max() - seg.min()
            path = np.sum(np.abs(np.diff(seg)))
            if path > 0 and price_range > 0:
                d = 1 + np.log(path / price_range) / np.log(window)
                fd[i] = np.clip(d, 1.0, 2.0)
            else:
                fd[i] = 1.5
        return pd.Series(fd, index=df.index)

    # ------------------------------------------------------------------
    # Market regime
    # ------------------------------------------------------------------
    def _detect_regime(self, df):
        up   = (df['SMA_20'] > df['SMA_50']).astype(int)
        down = (df['SMA_20'] < df['SMA_50']).astype(int)
        return (up - down).astype(float)

    # ------------------------------------------------------------------
    # Master indicator calculation
    # ------------------------------------------------------------------
    def calculate_indicators(self):
        df = self.df

        df['Returns']     = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Classic indicators
        df['RSI'] = self._rsi(df['Close'])
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD']        = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

        bm = df['Close'].rolling(20).mean()
        bs = df['Close'].rolling(20).std()
        df['BB_Mid']      = bm
        df['BB_Upper']    = bm + 2 * bs
        df['BB_Lower']    = bm - 2 * bs
        df['BB_Width']    = (df['BB_Upper'] - df['BB_Lower']) / bm.replace(0, np.nan)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)

        df['ATR'] = self._atr(df)
        df['OBV'] = self._obv(df)
        df['OBV_EMA'] = df['OBV'].ewm(span=20).mean()
        df['Stoch_K'], df['Stoch_D'] = self._stochastic(df)
        df['Williams_R'] = self._williams_r(df)
        df['CCI'] = self._cci(df)
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = self._adx(df)
        df['MFI'] = self._mfi(df)
        df['ROC'] = self._roc(df['Close'])

        df['Volume_SMA']   = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA'].replace(0, np.nan)

        for p in [5, 10, 20, 50, 200]:
            df[f'SMA_{p}'] = df['Close'].rolling(p).mean()
            df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20'].replace(0, np.nan)
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50'].replace(0, np.nan)

        # AEMI (novel)
        df['AEMI']        = self._adaptive_entropy_momentum_index(df)
        df['AEMI_Signal'] = df['AEMI'].ewm(span=9).mean()

        df['Regime']      = self._detect_regime(df)
        df['Fractal_Dim'] = self._fractal_dimension(df)

        # ── NEW INDICATORS ──────────────────────────────────────────
        # Hurst Exponent (the big one — detects trending vs mean-reverting)
        df['Hurst'] = self._hurst_exponent(df, window=100)

        # Z-Score Mean Reversion
        df['ZScore_20'] = self._zscore_mean_reversion(df, window=20)
        df['ZScore_50'] = self._zscore_mean_reversion(df, window=50)

        # Return Autocorrelation (lag-1 and lag-5)
        df['Autocorr_1'] = self._autocorrelation(df, lag=1, window=20)
        df['Autocorr_5'] = self._autocorrelation(df, lag=5, window=30)

        # Order Flow Imbalance
        df['OFI'] = self._order_flow_imbalance(df, window=14)

        # Multi-Timeframe Momentum Convergence
        df['MTF_Convergence'] = self._mtf_momentum_convergence(df)

        # Volatility Regime
        df['Vol_Regime'] = self._volatility_regime(df)

        # Volume Spike Z-Score
        df['Vol_Spike'] = self._volume_spike_zscore(df)

        # Range Ratio (candle body analysis)
        df['Range_Ratio'] = self._range_ratio(df)

        # Hurst-Adaptive Momentum: use Hurst to choose strategy
        # When Hurst > 0.5, follow momentum; when < 0.5, fade it
        raw_mom = df['EMA_10'] / df['EMA_20'].replace(0, np.nan) - 1
        hurst_centered = (df['Hurst'] - 0.5) * 2  # [-1, 1]
        df['Hurst_Adaptive_Mom'] = raw_mom * hurst_centered

        # Volatility-adjusted returns (Sharpe-like)
        df['VolAdj_Return_5'] = df['Returns'].rolling(5).mean() / df['Returns'].rolling(5).std().replace(0, np.nan)
        df['VolAdj_Return_20'] = df['Returns'].rolling(20).mean() / df['Returns'].rolling(20).std().replace(0, np.nan)

        self.df = df
        return df

    # ------------------------------------------------------------------
    # Feature matrix with selection
    # ------------------------------------------------------------------
    def prepare_features(self):
        df = self.df.copy()

        # Target: price higher N days later
        df['Target'] = (df['Close'].shift(-self.horizon) > df['Close']).astype(int)
        df['Future_Return'] = (df['Close'].shift(-self.horizon) / df['Close']) - 1

        base_features = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Width', 'BB_Position',
            'ATR', 'OBV_EMA',
            'Stoch_K', 'Stoch_D',
            'Williams_R', 'CCI',
            'ADX', 'MFI', 'ROC',
            'Volume_Ratio',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio',
            'AEMI', 'AEMI_Signal',
            'Regime', 'Fractal_Dim',
            'Returns', 'Log_Returns',
            # NEW indicators
            'Hurst',
            'ZScore_20', 'ZScore_50',
            'Autocorr_1', 'Autocorr_5',
            'OFI',
            'MTF_Convergence',
            'Vol_Regime',
            'Vol_Spike',
            'Range_Ratio',
            'Hurst_Adaptive_Mom',
            'VolAdj_Return_5', 'VolAdj_Return_20',
        ]

        # Lag features — focused on the most informative ones
        lag_features = []
        for col in ['RSI', 'MACD_Hist', 'AEMI', 'OFI', 'Hurst',
                     'ZScore_20', 'MTF_Convergence', 'Returns']:
            for lag in [1, 2, 3, 5]:
                name = f'{col}_lag{lag}'
                df[name] = df[col].shift(lag)
                lag_features.append(name)

        # Change features (momentum of indicators)
        change_features = []
        for col in ['RSI', 'MACD_Hist', 'Hurst', 'OFI', 'ADX']:
            name = f'{col}_change5'
            df[name] = df[col] - df[col].shift(5)
            change_features.append(name)

        # Rolling stats
        df['Return_Vol_5']   = df['Returns'].rolling(5).std()
        df['Return_Vol_10']  = df['Returns'].rolling(10).std()
        df['Return_Vol_20']  = df['Returns'].rolling(20).std()
        df['Return_Mean_5']  = df['Returns'].rolling(5).mean()
        df['Return_Mean_10'] = df['Returns'].rolling(10).mean()
        df['Return_Skew_20'] = df['Returns'].rolling(20).skew()
        df['Return_Kurt_20'] = df['Returns'].rolling(20).kurt()

        extra = ['Return_Vol_5', 'Return_Vol_10', 'Return_Vol_20',
                 'Return_Mean_5', 'Return_Mean_10', 'Return_Skew_20',
                 'Return_Kurt_20']

        all_features = base_features + lag_features + change_features + extra
        df_clean = df[all_features + ['Target', 'Future_Return', 'Close']].dropna()

        # Replace infinities
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

        return df_clean, all_features

    # ------------------------------------------------------------------
    # Feature selection via mutual information
    # ------------------------------------------------------------------
    def _select_features(self, X, y, feature_names, top_k=30):
        """
        Use mutual information to select the most predictive features.
        This removes noise features that hurt model performance.
        """
        mi_scores = mutual_info_classif(X, y, discrete_features=False,
                                         random_state=42, n_neighbors=5)
        mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

        # Keep features with MI > 0 up to top_k
        selected = mi_series[mi_series > 0].head(top_k).index.tolist()

        # Always keep our key novel indicators if they have any signal
        must_keep = ['Hurst', 'AEMI', 'Hurst_Adaptive_Mom', 'OFI',
                     'MTF_Convergence', 'ZScore_20']
        for feat in must_keep:
            if feat in feature_names and feat not in selected:
                if mi_series.get(feat, 0) > 0:
                    selected.append(feat)

        self.mi_scores = mi_series
        return selected

    # ------------------------------------------------------------------
    # Training with purged walk-forward CV and stacking
    # ------------------------------------------------------------------
    def train_model(self):
        df_clean, feature_cols = self.prepare_features()

        X_all_raw = df_clean[feature_cols].values
        y_all = df_clean['Target'].values

        # ── Feature selection on first 60% of data ──
        split_idx = int(len(X_all_raw) * 0.6)
        X_select = self.scaler.fit_transform(X_all_raw[:split_idx])
        y_select = y_all[:split_idx]
        selected = self._select_features(X_select, y_select, feature_cols, top_k=30)
        self.selected_features = selected

        # Rebuild with selected features only
        X = df_clean[selected].values
        y = y_all

        # ── Purged walk-forward CV ──
        # Add a gap of `horizon` days between train and test to prevent leakage
        n = len(X)
        n_splits = 5
        test_size = n // (n_splits + 1)
        gap = self.horizon

        cv_folds = []
        for i in range(n_splits):
            test_start = n - (n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - gap  # purge gap
            if train_end < test_size:
                continue
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n))
            cv_folds.append((train_idx, test_idx))

        if not cv_folds:
            # Fallback to standard TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_folds = list(tscv.split(X))

        # ── Models ──
        self.rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_split=30,
            min_samples_leaf=15,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

        if HAS_XGB:
            self.xgb_model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
            )
        else:
            self.xgb_model = None

        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )

        cv_scores = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []

        oos_predictions = np.full(n, np.nan)
        oos_probabilities = np.full(n, np.nan)

        for train_idx, test_idx in cv_folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            Xs_tr = self.scaler.fit_transform(X_tr)
            Xs_te = self.scaler.transform(X_te)

            # Train all base models
            self.rf_model.fit(Xs_tr, y_tr)
            self.gb_model.fit(Xs_tr, y_tr)

            rf_prob = self.rf_model.predict_proba(Xs_te)[:, 1]
            gb_prob = self.gb_model.predict_proba(Xs_te)[:, 1]

            if self.xgb_model is not None:
                self.xgb_model.fit(Xs_tr, y_tr)
                xgb_prob = self.xgb_model.predict_proba(Xs_te)[:, 1]
                # 3-model weighted ensemble (XGB gets more weight — it's the strongest)
                ensemble_prob = 0.35 * xgb_prob + 0.35 * rf_prob + 0.30 * gb_prob
            else:
                ensemble_prob = 0.5 * rf_prob + 0.5 * gb_prob

            ensemble_pred = (ensemble_prob >= 0.5).astype(int)

            oos_predictions[test_idx] = ensemble_pred
            oos_probabilities[test_idx] = ensemble_prob

            cv_scores.append(accuracy_score(y_te, ensemble_pred))
            cv_precision.append(precision_score(y_te, ensemble_pred, zero_division=0))
            cv_recall.append(recall_score(y_te, ensemble_pred, zero_division=0))
            cv_f1.append(f1_score(y_te, ensemble_pred, zero_division=0))

        # ── Final fit on all data ──
        X_all_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_all_scaled, y)
        self.gb_model.fit(X_all_scaled, y)
        if self.xgb_model is not None:
            self.xgb_model.fit(X_all_scaled, y)

        # ── Stacking meta-learner ──
        # Train a logistic regression on the base model probabilities
        meta_mask = ~np.isnan(oos_probabilities)
        if np.sum(meta_mask) > 50:
            rf_full_prob = self.rf_model.predict_proba(X_all_scaled)[:, 1]
            gb_full_prob = self.gb_model.predict_proba(X_all_scaled)[:, 1]
            if self.xgb_model is not None:
                xgb_full_prob = self.xgb_model.predict_proba(X_all_scaled)[:, 1]
                meta_X = np.column_stack([rf_full_prob, gb_full_prob, xgb_full_prob])
            else:
                meta_X = np.column_stack([rf_full_prob, gb_full_prob])

            self.meta_model = LogisticRegression(C=1.0, random_state=42)
            self.meta_model.fit(meta_X[meta_mask], y[meta_mask])

        self.feature_cols = feature_cols
        self.df_clean = df_clean

        # ── Build backtest ──
        backtest_mask = ~np.isnan(oos_predictions)
        bt_dates = df_clean.index[backtest_mask].strftime('%Y-%m-%d').tolist()
        bt_predictions = oos_predictions[backtest_mask].astype(int).tolist()
        bt_probabilities = oos_probabilities[backtest_mask].tolist()
        bt_actuals = y[backtest_mask].tolist()
        bt_returns = df_clean['Future_Return'].values[backtest_mask].tolist()
        bt_prices = df_clean['Close'].values[backtest_mask].tolist()

        bt_correct = [int(p == a) for p, a in zip(bt_predictions, bt_actuals)]
        rolling_window = min(20, max(5, len(bt_correct) // 10))
        rolling_acc = []
        for i in range(len(bt_correct)):
            start = max(0, i - rolling_window + 1)
            window = bt_correct[start:i + 1]
            rolling_acc.append(round(sum(window) / len(window) * 100, 1))

        # Equity curve
        equity = [100.0]
        for i in range(len(bt_predictions)):
            ret = bt_returns[i] if bt_returns[i] is not None and not np.isnan(bt_returns[i]) else 0
            if bt_predictions[i] == 1:
                equity.append(equity[-1] * (1 + ret))
            else:
                equity.append(equity[-1])
        equity = equity[1:]

        bh_equity = [100.0]
        for i in range(len(bt_returns)):
            ret = bt_returns[i] if bt_returns[i] is not None and not np.isnan(bt_returns[i]) else 0
            bh_equity.append(bh_equity[-1] * (1 + ret))
        bh_equity = bh_equity[1:]

        # Confidence-filtered backtest
        # Only count predictions where confidence > 55% either direction
        conf_threshold = 0.55
        conf_correct = 0
        conf_total = 0
        for i in range(len(bt_probabilities)):
            p = bt_probabilities[i]
            if p > conf_threshold or p < (1 - conf_threshold):
                conf_total += 1
                if bt_correct[i]:
                    conf_correct += 1

        all_pred = oos_predictions[backtest_mask].astype(int)
        all_actual = y[backtest_mask]
        cm = confusion_matrix(all_actual, all_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        streaks = self._compute_streaks(bt_correct)

        self.backtest = {
            'dates': bt_dates,
            'predictions': bt_predictions,
            'probabilities': [round(p * 100, 1) for p in bt_probabilities],
            'actuals': bt_actuals,
            'correct': bt_correct,
            'rolling_accuracy': rolling_acc,
            'prices': [round(p, 2) for p in bt_prices],
            'equity_curve': [round(e, 2) for e in equity],
            'bh_equity_curve': [round(e, 2) for e in bh_equity],
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn),
                'fp': int(fp), 'fn': int(fn),
            },
            'total_predictions': len(bt_predictions),
            'correct_predictions': sum(bt_correct),
            'conf_filtered_total': conf_total,
            'conf_filtered_correct': conf_correct,
            'conf_filtered_accuracy': round(conf_correct / conf_total * 100, 1) if conf_total > 0 else 0,
            'streaks': streaks,
        }

        self.cv_metrics = {
            'accuracy': [round(s * 100, 1) for s in cv_scores],
            'precision': [round(s * 100, 1) for s in cv_precision],
            'recall': [round(s * 100, 1) for s in cv_recall],
            'f1': [round(s * 100, 1) for s in cv_f1],
        }

        return float(np.mean(cv_scores)), [float(s) for s in cv_scores]

    def _compute_streaks(self, correct_list):
        if not correct_list:
            return {'max_win': 0, 'max_loss': 0, 'current': 0, 'current_type': 'none'}
        max_win = max_loss = streak = 0
        last_val = None
        for c in correct_list:
            if c == last_val:
                streak += 1
            else:
                streak = 1
                last_val = c
            if c == 1:
                max_win = max(max_win, streak)
            else:
                max_loss = max(max_loss, streak)
        current_val = correct_list[-1]
        current_streak = 0
        for c in reversed(correct_list):
            if c == current_val:
                current_streak += 1
            else:
                break
        return {
            'max_win': max_win, 'max_loss': max_loss,
            'current': current_streak,
            'current_type': 'win' if current_val == 1 else 'loss',
        }

    # ------------------------------------------------------------------
    # Prediction (with stacking meta-learner)
    # ------------------------------------------------------------------
    def predict(self):
        recent = self.df_clean.tail(90)
        X = self.scaler.transform(recent[self.selected_features].values)

        rf_prob = self.rf_model.predict_proba(X)[:, 1]
        gb_prob = self.gb_model.predict_proba(X)[:, 1]

        if self.xgb_model is not None:
            xgb_prob = self.xgb_model.predict_proba(X)[:, 1]

        # Use meta-learner if available
        if self.meta_model is not None:
            if self.xgb_model is not None:
                meta_X = np.column_stack([rf_prob, gb_prob, xgb_prob])
            else:
                meta_X = np.column_stack([rf_prob, gb_prob])
            ensemble = self.meta_model.predict_proba(meta_X)[:, 1]
        else:
            if self.xgb_model is not None:
                ensemble = 0.35 * xgb_prob + 0.35 * rf_prob + 0.30 * gb_prob
            else:
                ensemble = 0.5 * rf_prob + 0.5 * gb_prob

        dates = recent.index.strftime('%Y-%m-%d').tolist()
        return dates, ensemble.tolist(), rf_prob.tolist()

    def get_feature_importance(self):
        imp = pd.Series(
            self.rf_model.feature_importances_,
            index=self.selected_features,
        ).sort_values(ascending=False)
        return imp.head(15).to_dict()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self):
        self.fetch_data()
        self.calculate_indicators()
        accuracy, cv_scores = self.train_model()
        dates, predictions, rf_predictions = self.predict()

        latest = self.df.iloc[-1]
        feature_imp = self.get_feature_importance()
        recent_df = self.df.tail(120)
        current_signal = predictions[-1]

        def safe(val, decimals=2):
            v = float(val)
            return None if np.isnan(v) or np.isinf(v) else round(v, decimals)

        def clean_list(s, decimals=2):
            return [safe(x, decimals) for x in s]

        # Top MI scores for frontend
        mi_top = {}
        if hasattr(self, 'mi_scores'):
            mi_top = self.mi_scores.head(20).to_dict()

        return {
            'ticker':           self.ticker,
            'horizon':          self.horizon,
            'accuracy':         round(accuracy * 100, 1),
            'cv_scores':        [round(s * 100, 1) for s in cv_scores],
            'cv_metrics':       self.cv_metrics,
            'signal':           round(current_signal * 100, 1),
            'signal_direction': 'BULLISH' if current_signal > 0.5 else 'BEARISH',
            'signal_strength':  round(abs(current_signal - 0.5) * 200, 1),
            'current_price':    safe(latest['Close']),
            'rsi':              safe(latest['RSI'], 1),
            'macd':             safe(latest['MACD'], 4),
            'macd_signal':      safe(latest['MACD_Signal'], 4),
            'macd_hist':        safe(latest['MACD_Hist'], 4),
            'atr':              safe(latest['ATR']),
            'aemi':             safe(latest['AEMI'], 6),
            'bb_position':      safe(latest['BB_Position'] * 100, 1) if not np.isnan(latest['BB_Position']) else None,
            'volume_ratio':     safe(latest['Volume_Ratio']),
            'regime':           int(latest['Regime']) if not np.isnan(latest['Regime']) else 0,
            'fractal_dim':      safe(latest['Fractal_Dim'], 3),
            'stoch_k':          safe(latest['Stoch_K'], 1),
            'stoch_d':          safe(latest['Stoch_D'], 1),
            'williams_r':       safe(latest['Williams_R'], 1),
            'cci':              safe(latest['CCI'], 1),
            'adx':              safe(latest['ADX'], 1),
            'mfi':              safe(latest['MFI'], 1),
            'roc':              safe(latest['ROC'], 2),
            'hurst':            safe(latest['Hurst'], 3),
            'zscore':           safe(latest['ZScore_20'], 2),
            'ofi':              safe(latest['OFI'], 3),
            'mtf_convergence':  safe(latest['MTF_Convergence'], 2),
            'autocorr':         safe(latest['Autocorr_1'], 3),
            'vol_regime':       safe(latest['Vol_Regime'], 2),
            'dates':            dates,
            'predictions':      [round(p * 100, 1) for p in predictions],
            'feature_importance': feature_imp,
            'mi_scores':        mi_top,
            'selected_features': self.selected_features,
            'total_features':   len(self.feature_cols),
            'backtest':         self.backtest,
            'price_history': {
                'dates':    recent_df.index.strftime('%Y-%m-%d').tolist(),
                'close':    clean_list(recent_df['Close']),
                'sma20':    clean_list(recent_df['SMA_20']),
                'sma50':    clean_list(recent_df['SMA_50']),
                'bb_upper': clean_list(recent_df['BB_Upper']),
                'bb_lower': clean_list(recent_df['BB_Lower']),
                'volume':   [int(v) for v in recent_df['Volume']],
                'aemi':     clean_list(recent_df['AEMI'], 6),
                'rsi':      clean_list(recent_df['RSI'], 1),
                'macd':     clean_list(recent_df['MACD'], 4),
                'macd_signal': clean_list(recent_df['MACD_Signal'], 4),
                'macd_hist':   clean_list(recent_df['MACD_Hist'], 4),
                'stoch_k':  clean_list(recent_df['Stoch_K'], 1),
                'stoch_d':  clean_list(recent_df['Stoch_D'], 1),
                'adx':      clean_list(recent_df['ADX'], 1),
                'mfi':      clean_list(recent_df['MFI'], 1),
                'hurst':    clean_list(recent_df['Hurst'], 3),
            },
        }
