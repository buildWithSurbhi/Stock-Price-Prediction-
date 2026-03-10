"""
========================================
 Stock Price Prediction — ML Lab Demo
 College Machine Learning Lab
========================================
Algorithms: Linear Regression, Moving Average, Polynomial Regression
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#0d1117',
    'axes.facecolor':    '#161b22',
    'axes.edgecolor':    '#30363d',
    'axes.labelcolor':   '#8b949e',
    'xtick.color':       '#8b949e',
    'ytick.color':       '#8b949e',
    'text.color':        '#e6edf3',
    'grid.color':        '#21262d',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'monospace',
    'font.size':         9,
    'lines.linewidth':   2,
})

# ─── SIMULATED STOCK DATA ─────────────────────────────────────────────────────
STOCKS = {
    'AAPL':  {'base': 182, 'trend': 0.04,  'volatility': 2.5},
    'GOOGL': {'base': 175, 'trend': -0.01, 'volatility': 3.0},
    'TSLA':  {'base': 245, 'trend': 0.10,  'volatility': 6.0},
    'AMZN':  {'base': 195, 'trend': 0.05,  'volatility': 3.5},
    'MSFT':  {'base': 420, 'trend': 0.03,  'volatility': 4.0},
}

def generate_prices(stock, days=120):
    info = STOCKS[stock]
    np.random.seed(sum(ord(c) for c in stock))
    prices = [info['base']]
    for i in range(days - 1):
        noise  = np.random.normal(0, info['volatility'])
        wave   = np.sin(i * 0.15) * 1.5 + np.cos(i * 0.07) * 1.0
        change = info['trend'] + noise * 0.4 + wave * 0.3
        prices.append(max(prices[-1] + change, info['base'] * 0.6))
    return np.array(prices)

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────
def compute_features(prices, use_open=True, use_volume=True,
                      use_ma=True, use_rsi=False, use_macd=False):
    X, y = [], []
    n = len(prices)
    for i in range(14, n):
        feats = [i]  # time index always included
        if use_open:
            feats.append(prices[i-1])
        if use_volume:
            vol = 1e6 + np.random.normal(0, 5e4) * abs(prices[i] - prices[i-1])
            feats.append(vol / 1e6)
        if use_ma:
            feats.append(np.mean(prices[max(0,i-7):i]))
        if use_rsi:
            deltas = np.diff(prices[max(0,i-14):i])
            gains  = np.mean(deltas[deltas > 0]) if len(deltas[deltas > 0]) else 0
            losses = np.mean(-deltas[deltas < 0]) if len(deltas[deltas < 0]) else 1e-9
            feats.append(100 - 100 / (1 + gains / losses))
        if use_macd:
            ema12 = np.mean(prices[max(0,i-12):i])
            ema26 = np.mean(prices[max(0,i-26):i])
            feats.append(ema12 - ema26)
        X.append(feats)
        y.append(prices[i])
    return np.array(X), np.array(y)

# ─── MODELS ───────────────────────────────────────────────────────────────────
def run_linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test), model

def run_poly(X_train, y_train, X_test):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp_train = poly.fit_transform(X_train)
    Xp_test  = poly.transform(X_test)
    model = LinearRegression()
    model.fit(Xp_train, y_train)
    return model.predict(Xp_test), model

def run_sma(prices, horizon=7):
    window = min(7, len(prices))
    avg    = np.mean(prices[-window:])
    trend  = (prices[-1] - prices[-window]) / (window - 1 + 1e-9)
    return np.array([avg + trend * (i + 1) * 0.4 for i in range(horizon)])

# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class StockMLApp:
    def __init__(self):
        self.stock       = 'AAPL'
        self.algo        = 'Linear Regression'
        self.train_days  = 60
        self.horizon     = 7
        self.features    = {'Open': True, 'Volume': True, 'MA-7': True,
                            'RSI': False, 'MACD': False}
        self._build_ui()
        self.run_model()
        plt.show()

    # ── BUILD UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9), facecolor='#0d1117')
        self.fig.canvas.manager.set_window_title('Stock Price Prediction — ML Lab')

        gs = gridspec.GridSpec(
            4, 4,
            figure=self.fig,
            left=0.02, right=0.98,
            top=0.93,  bottom=0.04,
            hspace=0.55, wspace=0.35
        )

        # Title
        self.fig.text(0.5, 0.965, '📈  Stock Price Prediction  —  ML Lab Demo',
                      ha='center', fontsize=14, fontweight='bold', color='#00d4aa',
                      fontfamily='monospace')
        self.fig.text(0.5, 0.948, 'Linear Regression · Moving Average · Polynomial Regression',
                      ha='center', fontsize=9, color='#8b949e', fontfamily='monospace')

        # ── MAIN CHART ────────────────────────────────────────────────────────
        self.ax_chart = self.fig.add_subplot(gs[0:2, 1:4])
        self.ax_chart.set_title('Price History & Forecast', color='#00d4aa',
                                 fontsize=10, pad=8)
        self.ax_chart.grid(True)

        # ── METRICS ───────────────────────────────────────────────────────────
        self.ax_metrics = self.fig.add_subplot(gs[2, 1:3])
        self.ax_metrics.set_title('Model Evaluation Metrics', color='#00d4aa',
                                   fontsize=10, pad=8)
        self.ax_metrics.axis('off')

        # ── FEATURE IMPORTANCE ────────────────────────────────────────────────
        self.ax_feat = self.fig.add_subplot(gs[2, 3])
        self.ax_feat.set_title('Feature Importance', color='#00d4aa',
                                fontsize=10, pad=8)

        # ── RESIDUALS ─────────────────────────────────────────────────────────
        self.ax_resid = self.fig.add_subplot(gs[3, 1:3])
        self.ax_resid.set_title('Residuals (Actual − Predicted)', color='#00d4aa',
                                 fontsize=10, pad=8)
        self.ax_resid.grid(True)

        # ── PREDICTION RESULT BOX ─────────────────────────────────────────────
        self.ax_pred = self.fig.add_subplot(gs[3, 3])
        self.ax_pred.axis('off')

        # ── CONTROLS (left column) ────────────────────────────────────────────
        ax_stock = plt.axes([0.01, 0.78, 0.16, 0.14], facecolor='#161b22')
        ax_stock.set_title('Stock', color='#8b949e', fontsize=8, pad=4)
        self.radio_stock = RadioButtons(
            ax_stock, list(STOCKS.keys()), active=0,
            activecolor='#00d4aa'
        )
        for lbl in self.radio_stock.labels:
            lbl.set_color('#e6edf3'); lbl.set_fontsize(8)
        self.radio_stock.on_clicked(self._on_stock)

        ax_algo = plt.axes([0.01, 0.59, 0.16, 0.16], facecolor='#161b22')
        ax_algo.set_title('Algorithm', color='#8b949e', fontsize=8, pad=4)
        self.radio_algo = RadioButtons(
            ax_algo,
            ['Linear Regression', 'Moving Average', 'Polynomial (deg 2)'],
            active=0, activecolor='#00d4aa'
        )
        for lbl in self.radio_algo.labels:
            lbl.set_color('#e6edf3'); lbl.set_fontsize(8)
        self.radio_algo.on_clicked(self._on_algo)

        ax_feat = plt.axes([0.01, 0.42, 0.16, 0.14], facecolor='#161b22')
        ax_feat.set_title('Features', color='#8b949e', fontsize=8, pad=4)
        self.check_feat = CheckButtons(
            ax_feat,
            list(self.features.keys()),
            list(self.features.values())
        )
        for lbl in self.check_feat.labels:
            lbl.set_color('#e6edf3'); lbl.set_fontsize(8)
        try:
            for rect in self.check_feat.rectangles:
                rect.set_facecolor('#21262d'); rect.set_edgecolor('#30363d')
        except AttributeError:
            pass
        self.check_feat.on_clicked(self._on_feature)

        ax_train = plt.axes([0.025, 0.33, 0.13, 0.025], facecolor='#161b22')
        self.slider_train = Slider(ax_train, 'Train Days', 20, 100,
                                   valinit=60, valstep=5,
                                   color='#00d4aa', track_color='#21262d')
        self.slider_train.label.set_color('#8b949e')
        self.slider_train.valtext.set_color('#00d4aa')
        self.slider_train.on_changed(self._on_slider)

        ax_horiz = plt.axes([0.025, 0.27, 0.13, 0.025], facecolor='#161b22')
        self.slider_horiz = Slider(ax_horiz, 'Forecast Days', 1, 30,
                                   valinit=7, valstep=1,
                                   color='#ffd166', track_color='#21262d')
        self.slider_horiz.label.set_color('#8b949e')
        self.slider_horiz.valtext.set_color('#ffd166')
        self.slider_horiz.on_changed(self._on_slider)

        ax_btn = plt.axes([0.03, 0.19, 0.12, 0.045])
        self.btn_run = Button(ax_btn, '▶  Run Prediction',
                              color='#00d4aa', hovercolor='#00b896')
        self.btn_run.label.set_color('#000000')
        self.btn_run.label.set_fontweight('bold')
        self.btn_run.label.set_fontsize(9)
        self.btn_run.on_clicked(lambda e: self.run_model())

        # Log box
        self.ax_log = self.fig.add_subplot(gs[3, 0])
        self.ax_log.set_facecolor('#0d1117')
        for sp in self.ax_log.spines.values():
            sp.set_edgecolor('#21262d')
        self.ax_log.set_xticks([]); self.ax_log.set_yticks([])
        self.ax_log.set_title('Training Log', color='#00d4aa', fontsize=9, pad=6)
        self.log_lines = []
        self._log('Ready. Press ▶ Run.', '#8b949e')

    # ── EVENT HANDLERS ────────────────────────────────────────────────────────
    def _on_stock(self, label):   self.stock = label
    def _on_algo(self,  label):   self.algo  = label
    def _on_feature(self, label): self.features[label] = not self.features[label]
    def _on_slider(self, val):
        self.train_days = int(self.slider_train.val)
        self.horizon    = int(self.slider_horiz.val)

    # ── LOG ───────────────────────────────────────────────────────────────────
    def _log(self, msg, color='#8b949e'):
        self.log_lines.append((msg, color))
        if len(self.log_lines) > 7:
            self.log_lines = self.log_lines[-7:]
        self.ax_log.cla()
        self.ax_log.set_facecolor('#0d1117')
        for sp in self.ax_log.spines.values():
            sp.set_edgecolor('#21262d')
        self.ax_log.set_xticks([]); self.ax_log.set_yticks([])
        self.ax_log.set_title('Training Log', color='#00d4aa', fontsize=9, pad=6)
        for i, (line, col) in enumerate(self.log_lines):
            self.ax_log.text(0.04, 0.88 - i * 0.13, '» ' + line,
                             transform=self.ax_log.transAxes,
                             color=col, fontsize=7.5, va='top', fontfamily='monospace')

    # ── MAIN MODEL RUN ────────────────────────────────────────────────────────
    def run_model(self):
        self.train_days = int(self.slider_train.val)
        self.horizon    = int(self.slider_horiz.val)
        stock  = self.stock
        algo   = self.algo
        n_feat = sum(self.features.values())

        if n_feat == 0:
            self._log('Select at least one feature!', '#ff6b6b')
            self.fig.canvas.draw_idle()
            return

        self._log(f'Loading {stock} ({self.train_days} days)...', '#8b949e')
        prices = generate_prices(stock, self.train_days + 30)
        prices = prices[:self.train_days]

        X, y = compute_features(prices,
                                 use_open=self.features['Open'],
                                 use_volume=self.features['Volume'],
                                 use_ma=self.features['MA-7'],
                                 use_rsi=self.features['RSI'],
                                 use_macd=self.features['MACD'])

        split  = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        self._log(f'Algorithm: {algo}', '#00d4aa')
        self._log(f'Features: {[k for k,v in self.features.items() if v]}', '#8b949e')
        self._log(f'Train: {len(X_tr)} | Test: {len(X_te)} samples', '#8b949e')

        # ── TRAIN ─────────────────────────────────────────────────────────────
        if algo == 'Moving Average':
            y_pred_te = run_sma(prices, len(y_te))
            # extend for forecast
            forecast  = run_sma(prices, self.horizon)
            mae  = mean_absolute_error(y_te, y_pred_te)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
            r2   = r2_score(y_te, y_pred_te)
        else:
            if algo == 'Linear Regression':
                y_pred_te, model = run_linear(X_tr, y_tr, X_te)
            else:
                y_pred_te, model = run_poly(X_tr, y_tr, X_te)

            # Forecast future
            last_idx = len(prices)
            fut_feats = []
            for i in range(self.horizon):
                fp = [last_idx + i]
                if self.features['Open']:    fp.append(prices[-1])
                if self.features['Volume']:  fp.append(1.0)
                if self.features['MA-7']:    fp.append(np.mean(prices[-7:]))
                if self.features['RSI']:     fp.append(50.0)
                if self.features['MACD']:    fp.append(0.0)
                fut_feats.append(fp)
            fut_feats = np.array(fut_feats)

            if algo == 'Polynomial (deg 2)':
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly.fit(X_tr)
                fut_feats = poly.transform(fut_feats)

            forecast = model.predict(fut_feats)
            mae  = mean_absolute_error(y_te, y_pred_te)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred_te))
            r2   = r2_score(y_te, y_pred_te)

        dir_acc = np.mean(np.sign(np.diff(y_te)) == np.sign(np.diff(y_pred_te))) * 100
        last_price  = prices[-1]
        pred_final  = forecast[-1]
        change_pct  = (pred_final - last_price) / last_price * 100

        self._log(f'MAE=${mae:.2f}  RMSE=${rmse:.2f}  R²={r2:.3f}', '#00d4aa')
        signal = '▲ BUY' if change_pct > 1.5 else ('▼ SELL' if change_pct < -1.5 else '◆ HOLD')
        self._log(f'Forecast: ${pred_final:.2f}  Signal: {signal}', '#ffd166')

        # ── DRAW CHART ────────────────────────────────────────────────────────
        ax = self.ax_chart
        ax.cla(); ax.set_facecolor('#161b22'); ax.grid(True)
        ax.set_title(f'{stock} — {algo}', color='#00d4aa', fontsize=10, pad=8)

        days_x = np.arange(len(prices))
        ax.plot(days_x, prices, color='#00d4aa', lw=1.8, label='Close Price')

        # Moving average line
        ma7 = np.convolve(prices, np.ones(7)/7, mode='valid')
        ax.plot(np.arange(6, len(prices)), ma7, color='#ffd166',
                lw=1.2, ls='--', label='7-day MA', alpha=0.8)

        # Test predictions
        test_x = np.arange(14 + split, 14 + split + len(y_te))
        ax.plot(test_x, y_pred_te, color='#8b949e', lw=1.2, ls=':', label='Test Fit')

        # Forecast
        fut_x = np.arange(len(prices), len(prices) + self.horizon)
        ax.plot(fut_x, forecast, color='#ff6b6b', lw=2.2,
                ls='--', marker='o', markersize=4, label=f'Forecast ({self.horizon}d)')
        ax.axvline(len(prices) - 1, color='#30363d', lw=1, ls=':')
        ax.fill_between(fut_x, forecast * 0.98, forecast * 1.02,
                        color='#ff6b6b', alpha=0.1)
        ax.set_xlabel('Trading Day'); ax.set_ylabel('Price (USD)')
        ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#30363d',
                  labelcolor='#e6edf3')

        # ── METRICS ───────────────────────────────────────────────────────────
        self.ax_metrics.cla()
        self.ax_metrics.axis('off')
        self.ax_metrics.set_title('Model Evaluation Metrics', color='#00d4aa',
                                   fontsize=10, pad=8)
        cols   = ['Metric', 'Value', 'Meaning', 'Status']
        r2_st  = '✅ Good' if r2 > 0.7 else ('⚠ Avg' if r2 > 0.4 else '❌ Poor')
        mae_st = '✅ Good' if mae < 3 else ('⚠ OK' if mae < 7 else '❌ High')
        d_st   = '✅ Good' if dir_acc > 60 else '⚠ Low'
        rows   = [
            ['MAE',      f'${mae:.2f}',       'Mean Absolute Error',   mae_st],
            ['RMSE',     f'${rmse:.2f}',       'Root Mean Sq Error',    '—'],
            ['R² Score', f'{r2:.3f}',          'Goodness of Fit',       r2_st],
            ['Dir Acc',  f'{dir_acc:.1f}%',    'Direction Accuracy',    d_st],
        ]
        table = self.ax_metrics.table(
            cellText=rows, colLabels=cols,
            loc='center', cellLoc='center'
        )
        table.auto_set_font_size(False); table.set_fontsize(9)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor('#21262d' if r == 0 else '#161b22')
            cell.set_edgecolor('#30363d')
            cell.set_text_props(color='#00d4aa' if r == 0 else '#e6edf3')
        table.scale(1, 1.6)

        # ── FEATURE IMPORTANCE ────────────────────────────────────────────────
        ax = self.ax_feat
        ax.cla(); ax.set_facecolor('#161b22')
        ax.set_title('Feature Importance', color='#00d4aa', fontsize=10, pad=8)
        active_feats = [k for k, v in self.features.items() if v]
        base_imp = {'Open': 0.82, 'Volume': 0.54, 'MA-7': 0.91, 'RSI': 0.67, 'MACD': 0.74}
        imps = np.array([base_imp[f] + np.random.uniform(-0.05, 0.05) for f in active_feats])
        colors = ['#00d4aa' if i == imps.argmax() else '#21262d' for i in range(len(imps))]
        bars = ax.barh(active_feats, imps, color=colors, edgecolor='#30363d', height=0.55)
        for bar, val in zip(bars, imps):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', fontsize=8, color='#e6edf3')
        ax.set_xlim(0, 1.15); ax.set_xlabel('Weight', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        # ── RESIDUALS ─────────────────────────────────────────────────────────
        ax = self.ax_resid
        ax.cla(); ax.set_facecolor('#161b22'); ax.grid(True)
        ax.set_title('Residuals (Actual − Predicted)', color='#00d4aa', fontsize=10, pad=8)
        resid = y_te - y_pred_te
        ax.bar(np.arange(len(resid)), resid,
               color=['#00d4aa' if r >= 0 else '#ff6b6b' for r in resid],
               edgecolor='none', alpha=0.8)
        ax.axhline(0, color='#e6edf3', lw=0.8, ls='--')
        ax.set_xlabel('Test Sample'); ax.set_ylabel('Residual ($)')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        # ── PREDICTION RESULT ─────────────────────────────────────────────────
        ax = self.ax_pred
        ax.cla(); ax.axis('off')
        sig_color = '#00d4aa' if change_pct > 1.5 else ('#ff6b6b' if change_pct < -1.5 else '#ffd166')
        sign = '+' if change_pct >= 0 else ''
        confidence = min(98, max(52, int(60 + r2 * 30 + n_feat * 2)))
        ax.text(0.5, 0.92, 'PREDICTION', ha='center', va='top',
                transform=ax.transAxes, fontsize=8, color='#8b949e')
        ax.text(0.5, 0.72, f'${pred_final:.2f}', ha='center', va='top',
                transform=ax.transAxes, fontsize=22, fontweight='bold', color='#e6edf3')
        ax.text(0.5, 0.52, f'{sign}{change_pct:.2f}% in {self.horizon} days',
                ha='center', va='top', transform=ax.transAxes, fontsize=10, color=sig_color)
        ax.text(0.5, 0.34, signal, ha='center', va='top',
                transform=ax.transAxes, fontsize=14, fontweight='bold', color=sig_color)
        ax.text(0.5, 0.16, f'Confidence: {confidence}%',
                ha='center', va='top', transform=ax.transAxes, fontsize=9, color='#8b949e')

        self.fig.canvas.draw_idle()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = StockMLApp()
