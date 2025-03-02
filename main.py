import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='trend_indicator_debug.log'
)

class TrendIndicatorA:
    def __init__(self, stock_code="000980.SS", start_date="2023-06-01", end_date="2025-02-28", ma_period=15, trend_threshold=10, require_qqe_red_on_sell=True):
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.ma_period = ma_period
        self.trend_threshold = trend_threshold
        self.require_qqe_red_on_sell = require_qqe_red_on_sell
        self.previous_color = None
        self.changes = []
        self.transactions = []
        self.position = 0
        self.initial_funds = 100000
        self.current_funds = self.initial_funds
        self.last_trend_change_date = None
        self.last_trend_change_to = None
        self.transaction_fee_rate = 0.002
        self.stop_loss_percent = 0.10

        logging.info(f"Initializing with stock_code: {stock_code}, start_date: {start_date}, end_date: {end_date}, require_qqe_red_on_sell: {require_qqe_red_on_sell}")
        
        self.data = self._get_stock_data()
        if self.data is not None:
            logging.debug(f"Stock data shape: {self.data.shape}")
            self.heikin_ashi = self._calculate_heikin_ashi()
            self.trend = self._calculate_trend()
            self.qqe_colors = self._calculate_qqe_mod()

    def _get_stock_data(self):
        """使用 yfinance 获取股票数据"""
        ticker = yf.Ticker(self.stock_code)
        data = ticker.history(start=self.start_date, end=self.end_date, interval="1d")
        
        if data.empty:
            logging.warning(f"No stock data retrieved for {self.stock_code}")
            return None

        # 调整列名以匹配原有代码
        data = data.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        data.index = pd.to_datetime(data.index).tz_localize(None)  # 移除时区信息
        data["code"] = self.stock_code
        return data[["code", "open", "high", "low", "close", "volume"]]

    def _calculate_heikin_ashi(self):
        ha_close = (self.data["open"] + self.data["high"] + self.data["low"] + self.data["close"]) / 4
        ha_open = np.zeros(len(self.data))
        ha_open[0] = (self.data["open"].iloc[0] + self.data["close"].iloc[0]) / 2
        for i in range(1, len(self.data)):
            ha_open[i] = (ha_close.iloc[i - 1] + ha_open[i - 1]) / 2
        ha_high = self.data[["open", "close", "high"]].max(axis=1)
        ha_low = self.data[["open", "close", "low"]].min(axis=1)
        return pd.DataFrame({"Open": ha_open, "Close": ha_close, "High": ha_high, "Low": ha_low}, index=self.data.index)

    def _calculate_trend(self):
        ma_close = self.heikin_ashi["Close"].ewm(span=self.ma_period, adjust=False).mean()
        ma_open = self.heikin_ashi["Open"].ewm(span=self.ma_period, adjust=False).mean()
        ha_high = self.heikin_ashi["High"]
        ha_low = self.heikin_ashi["Low"]
        trend = 100 * (ma_close - ma_open) / (ha_high - ha_low)
        return trend.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _calculate_rsi(self, prices, period):
        """手动实现 RSI"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[1:period + 1])
        avg_loss = np.mean(loss[1:period + 1])

        rsi = [np.nan] * period
        rsi.append(100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100)

        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            rsi.append(100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100)

        return np.array(rsi)

    def _calculate_ema(self, data, period):
        """手动实现 EMA"""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def _calculate_qqe(self, prices, rsi_length, smoothing_factor, qqe_factor):
        rsi = self._calculate_rsi(prices, rsi_length)
        smoothed_rsi = self._calculate_ema(rsi, smoothing_factor)
        atr_rsi = np.abs(np.diff(smoothed_rsi))
        atr_rsi = np.concatenate([[np.nan], atr_rsi])
        wilders_length = rsi_length * 2 - 1
        smoothed_atr_rsi = self._calculate_ema(atr_rsi, wilders_length)
        dynamic_atr_rsi = smoothed_atr_rsi * qqe_factor

        long_band = np.zeros_like(smoothed_rsi)
        short_band = np.zeros_like(smoothed_rsi)
        trend_direction = np.zeros_like(smoothed_rsi)

        for i in range(1, len(smoothed_rsi)):
            if pd.isna(smoothed_rsi[i]):
                continue
            atr_delta = dynamic_atr_rsi[i]
            new_short_band = smoothed_rsi[i] + atr_delta
            new_long_band = smoothed_rsi[i] - atr_delta
            if smoothed_rsi[i-1] > long_band[i-1] and smoothed_rsi[i] > long_band[i-1]:
                long_band[i] = max(long_band[i-1], new_long_band)
            else:
                long_band[i] = new_long_band
            if smoothed_rsi[i-1] < short_band[i-1] and smoothed_rsi[i] < short_band[i-1]:
                short_band[i] = min(short_band[i-1], new_short_band)
            else:
                short_band[i] = new_short_band
            if smoothed_rsi[i] > short_band[i-1]:
                trend_direction[i] = 1
            elif smoothed_rsi[i-1] > long_band[i-1] and smoothed_rsi[i] < long_band[i-1]:
                trend_direction[i] = -1
            else:
                trend_direction[i] = trend_direction[i-1]

        qqe_trend_line = np.where(trend_direction == 1, long_band, short_band)
        return qqe_trend_line, smoothed_rsi

    def _calculate_qqe_mod(self):
        rsi_length_primary = 6
        rsi_smoothing_primary = 5
        qqe_factor_primary = 3.0
        rsi_length_secondary = 6
        rsi_smoothing_secondary = 5
        qqe_factor_secondary = 1.61
        threshold_secondary = 3.0
        bollinger_length = 50
        bollinger_multiplier = 0.35

        prices = self.data['close'].values
        primary_qqe, primary_rsi = self._calculate_qqe(prices, rsi_length_primary, rsi_smoothing_primary, qqe_factor_primary)
        secondary_qqe, secondary_rsi = self._calculate_qqe(prices, rsi_length_secondary, rsi_smoothing_secondary, qqe_factor_secondary)

        qqe_normalized = primary_qqe - 50
        bollinger_basis = pd.Series(qqe_normalized).rolling(bollinger_length).mean()
        bollinger_std = pd.Series(qqe_normalized).rolling(bollinger_length).std()
        bollinger_upper = bollinger_basis + bollinger_multiplier * bollinger_std
        bollinger_lower = bollinger_basis - bollinger_multiplier * bollinger_std

        rsi_primary_normalized = primary_rsi - 50
        rsi_secondary_normalized = secondary_rsi - 50

        colors = []
        for i in range(len(prices)):
            if pd.isna(rsi_primary_normalized[i]):
                colors.append('gray')
                continue
            if rsi_primary_normalized[i] > bollinger_upper[i]:
                color = '#00c3ff'
            elif rsi_primary_normalized[i] < bollinger_lower[i]:
                color = '#ff0062'
            else:
                color = '#707070'
            if rsi_secondary_normalized[i] > threshold_secondary and rsi_primary_normalized[i] > bollinger_upper[i]:
                color = '#00c3ff'
            elif rsi_secondary_normalized[i] < -threshold_secondary and rsi_primary_normalized[i] < bollinger_lower[i]:
                color = '#ff0062'
            colors.append(color)

        return pd.Series(colors, index=self.data.index)

    def _is_within_n_trading_days(self, current_date, trading_days, n=3):
        if self.last_trend_change_date is None:
            return False
        try:
            change_idx = trading_days.get_loc(self.last_trend_change_date)
            current_idx = trading_days.get_loc(current_date)
            return 1 <= current_idx - change_idx <= n
        except KeyError:
            return False

    def _force_sell(self, time_point, sell_price):
        if self.position > 0:
            quantity = self.position
            proceeds = sell_price * quantity * (1 - self.transaction_fee_rate)
            self.current_funds += proceeds
            self.position = 0
            self.transactions.append({'time': time_point, 'action': 'stop_loss_sell', 'price': sell_price, 'quantity': quantity, 'proceeds': proceeds})

    def check_trend_changes(self):
        current_color = None
        previous_color = None
        trend_confirmation_dates = []
        trading_days = self.data.index

        for i, time_point in enumerate(trading_days):
            value = self.trend.loc[time_point]
            current_color = 'green' if value > self.trend_threshold else 'red' if value < -self.trend_threshold else previous_color or 'red'

            try:
                current_price = self.data.loc[time_point]
                qqe_color = self.qqe_colors.loc[time_point]
                if self.position > 0 and self.transactions and self.transactions[-1]['action'] == 'buy':
                    buy_price = self.transactions[-1]['price']
                    if current_price['close'] < buy_price * (1 - self.stop_loss_percent):
                        self._force_sell(time_point, current_price['close'])
                        continue

                if previous_color is not None:
                    if current_color == previous_color:
                        trend_confirmation_dates.append(time_point)
                    else:
                        trend_confirmation_dates = [time_point]
                        self.last_trend_change_date = time_point
                        self.last_trend_change_to = current_color
                    if len(trend_confirmation_dates) >= 2 and current_color != previous_color:
                        self._execute_trade(time_point, previous_color, current_color, current_price, qqe_color)
                    elif self._is_within_n_trading_days(time_point, trading_days, n=3) and len(trend_confirmation_dates) >= 1:
                        if self.last_trend_change_to == 'green' and self.position == 0:
                            self._execute_trade(time_point, 'red', 'green', current_price, qqe_color)
                        elif self.last_trend_change_to == 'red' and self.position > 0:
                            self._execute_trade(time_point, 'green', 'red', current_price, qqe_color)

            except KeyError as e:
                logging.error(f"KeyError at {time_point}: {e}")

            previous_color = current_color

    def _execute_trade(self, time_point, from_color, to_color, current_price, qqe_color):
        if (from_color == 'red' and to_color == 'green' and qqe_color == '#00c3ff' and self.position == 0):
            buy_price = current_price['close']
            quantity = int(self.current_funds / buy_price / 100) * 100
            if quantity > 0:
                cost = buy_price * quantity * (1 + self.transaction_fee_rate)
                if cost <= self.current_funds + 1000:
                    self.position += quantity
                    self.current_funds -= cost
                    self.transactions.append({'time': time_point, 'action': 'buy', 'price': buy_price, 'quantity': quantity, 'cost': cost})
        elif (from_color == 'green' and to_color == 'red' and self.position > 0 and 
              (not self.require_qqe_red_on_sell or qqe_color == '#ff0062')):
            sell_price = current_price['close']
            quantity = self.position
            proceeds = sell_price * quantity * (1 - self.transaction_fee_rate)
            self.current_funds += proceeds
            self.position = 0
            self.transactions.append({'time': time_point, 'action': 'sell', 'price': sell_price, 'quantity': quantity, 'proceeds': proceeds})

    def check_recent_buy_signal(self, days=3):
        """检查最近days个交易日内是否有买入信号"""
        if self.data is None or len(self.data) < 2:
            return None
        
        trading_days = self.data.index
        recent_days = trading_days[-days:]
        
        current_color = None
        previous_color = None
        trend_confirmation_dates = []

        for i, time_point in enumerate(trading_days):
            value = self.trend.loc[time_point]
            current_color = 'green' if value > self.trend_threshold else 'red' if value < -self.trend_threshold else previous_color or 'red'

            if time_point in recent_days:
                try:
                    qqe_color = self.qqe_colors.loc[time_point]
                    if previous_color == 'red' and current_color == 'green' and qqe_color == '#00c3ff':
                        return time_point
                except KeyError:
                    continue

            if previous_color is not None:
                if current_color == previous_color:
                    trend_confirmation_dates.append(time_point)
                else:
                    trend_confirmation_dates = [time_point]
            previous_color = current_color

        return None

def get_all_stocks():
    """预定义部分中国A股股票代码（Yahoo Finance 不提供直接获取所有股票的 API）"""
    # 这里仅列出部分示例股票代码，可根据需要扩展
    stock_list = [
        "000001.SZ",  # 平安银行
        "000002.SZ",  # 万科A
        "600000.SS",  # 浦发银行
        "600519.SS",  # 贵州茅台
        "601318.SS",  # 中国平安
    ]
    return stock_list

def screen_stocks_for_buy_signals(start_date="2024-06-01", end_date="2025-02-28", days=3):
    """筛选股票并检查最近days天内的买入信号"""
    stock_list = get_all_stocks()
    buy_signals = []

    for stock_code in stock_list[:100]:  # 限制为前100只股票，可根据需要调整
        try:
            logging.info(f"Processing stock: {stock_code}")
            indicator = TrendIndicatorA(
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                trend_threshold=10,
                require_qqe_red_on_sell=True
            )
            buy_signal_date = indicator.check_recent_buy_signal(days=days)
            if buy_signal_date:
                buy_signals.append({"stock_code": stock_code, "buy_signal_date": buy_signal_date})
                print(f"Stock {stock_code} has a buy signal on {buy_signal_date}")
        except Exception as e:
            logging.error(f"Error processing {stock_code}: {str(e)}")
            continue

    return buy_signals

if __name__ == "__main__":
    try:
        logging.info("Starting stock screening")
        buy_signals = screen_stocks_for_buy_signals(
            start_date="2024-06-01",
            end_date="2025-02-28",
            days=3
        )

        print("\nStocks with Buy Signals in Last 3 Trading Days:")
        for signal in buy_signals:
            print(f"Stock: {signal['stock_code']}, Buy Signal Date: {signal['buy_signal_date']}")

    except Exception as e:
        logging.error(f"Main execution error: {str(e)}", exc_info=True)
        print(f"Error: {e}")
