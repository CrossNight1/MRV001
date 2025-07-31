from algo_deploy.strategy_module import LiveStrategy, Indicator, extract_ochlv, Indicator_2
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import signal
import time

class MRV001(LiveStrategy):
    """
    MRV001: Mean Reversion Strategy - v001 - Kalman -> Z-score -> PCA -> arbitrage oportunities.
    """

    def __init__(self, strategy_name: str, params: Dict, account_id: str, exchange: str, base: str, quote: str,
                 category: str = 'futures', demo: bool = True):
        super().__init__(strategy_name, account_id, exchange, base, quote, category, demo)
        self.add_params(params)
        self.stop_flag = False

        # Setup signal handler once on init
        signal.signal(signal.SIGINT, self._signal_handler)

        self.init_balance: float = 0
        self.placed_orders: Dict[int, str] = {} 
        self._cached_candles: Dict[str, Dict[str, np.array]] = {}
        self._last_fetch: float = 0

    def add_params(self, params: dict):
        """Update strategy parameters."""
        # --- Core strategy logic ---
        self.pairs = params.get('pairs', [['BTC_USDT', 'ETH_USDT'], ['DOGE_USDT', 'SHIB1000_USDT'], ['SAND_USDT', 'MANA_USDT']])
        self.window = params.get('window', 30)
        self.z_threshold = params.get('z_threshold', 1)
        self.value_entry = params.get('value_entry', 10)
        self.stop_loss_threshold = params.get('stop_loss_threshold', 0.2)

        # --- Execution / Order parameters ---
        self.max_position = params.get('max_position', 10)
        self.refresh_order = params.get('refresh_order', 60)

        # --- Time settings ---
        self.sleep_time = params.get('sleep_time', 60)

    def _extract_symbol(self, symbol: str):
        """Extract symbol from base and quote."""
        all_symbols = [symbol for pair in self.pairs for symbol in pair]
        if symbol not in all_symbols:
            raise ValueError(f"{symbol} is not supported.")
        return symbol.split('_')

    def _extract_all_candles(self):
        """Extract all candles for all symbols with caching and error handling."""
        # Cache results to avoid excessive API calls
        now = time.time()
        if hasattr(self, "_cached_candles") and (now - getattr(self, "_last_fetch", 0) < 10):
            return self._cached_candles

        result = {}
        all_symbols = [symbol for pair in self.pairs for symbol in pair]

        for symbol in all_symbols:
            try:
                base, quote = self._extract_symbol(symbol)
                candles = self.get_candles(base, quote, interval="1m", use_redis=False)

                if not candles or len(candles) == 0:
                    self.logger.warning(f"No candle data available for {symbol}, skipping.")
                    continue

                o, c, h, l, v = extract_ochlv(candles)
                if len(c) == 0:
                    self.logger.warning(f"Empty close prices for {symbol}, skipping.")
                    continue

                result[symbol] = {'o': o, 'c': c, 'h': h, 'l': l, 'v': v}

            except Exception as e:
                self.logger.error(f"Error extracting candles for {symbol}: {e}")
                continue

        # Cache fetched data
        self._cached_candles = result
        self._last_fetch = now

        return result

    def _price_df(self, pair: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute aligned price and returns DataFrame for all symbols."""
        price = pd.DataFrame()
        candles_dict = self._extract_all_candles()

        for symbol, candle_data in candles_dict.items():
            if symbol not in pair:
                continue
            c = candle_data['c']
            if isinstance(c, (list, np.ndarray)):
                c = pd.Series(c)
            price[symbol] = c

        price = price.dropna(how='all')

        return price

    def pair_signal(self, pair: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate entry signals using a single global threshold for this pair.
        """
        window = self.window
        curr_z = np.zeros(len(pair))
        curr_signal = np.zeros(len(pair))
        curr_vola = np.zeros(len(pair))

        try:
            prices_df = self._price_df(pair)
            if prices_df.shape[0] < self.window:
                self.logger.info(f"Not enough data for pair {pair}.")
                return curr_signal, curr_vola

            prices = prices_df.values
            T, N = prices.shape

            # Kalman mean
            kalman_prices = np.zeros_like(prices)
            for i in range(N):
                kalman_prices[:, i] = Indicator.kalman_filter_mean(prices[:, i], adjust_factor=0.0001)

            # Z-score
            z_kalman = np.zeros_like(prices)
            for i in range(N):
                series = prices[:, i]
                std = Indicator.moving_average(series, 5, "EMA")
                std[std < 1e-6] = 1e-6
                mean = kalman_prices[:, i]
                z_kalman[:, i] = (series - mean) / std

            # PCA basket
            weights = Indicator_2.pca_basket_weights(z_kalman[-window:])
            basket_z = Indicator_2.basket_prices(z_kalman[-window:], weights)

            if len(basket_z) < window:
                window = len(basket_z)

            # Global threshold
            _, z, signals = Indicator_2.compute_all_signals(
                z_kalman[-window - 1:],
                basket_z[-window - 1:],
                window=window,
                threshold=self.z_threshold,
                adj_factors=0.0001
            )

            curr_signal = signals[-1]
            curr_z = z[-1]

            rets = prices_df.pct_change().dropna()
            for i in range(N):
                curr_vola[i] = Indicator.rolling_std(rets.iloc[:, i].values, window)[-1]
                
            self.logger.info(
                f"Pair {pair} → "
                f"Signals: {curr_signal.tolist()} | "
                f"Z-score: {[round(x, 4) for x in curr_z.tolist()]} | "
                f"Vola: {[round(x, 4) for x in curr_vola.tolist()]}"
            )

        except Exception as e:
            self.logger.error(f"Error in pair_signal ({pair}): {e} | Line: {e.__traceback__.tb_lineno}")

        return curr_signal, curr_vola

    def get_entry(self, pair: list[str]) -> Dict[str, tuple[int, float]]:
        """
        Generate entry signals for available symbols using Z-score + Order Book Imbalance filter.
        Returns:
            Dict[symbol -> (signal, price_limit)]
        """
        signal, vola = self.pair_signal(pair)
        if not np.any(signal):
            return {}

        entry_dict = {}
        open_positions = self.get_opened_positions()
        obi_threshold = 2  # constant threshold
        order_results = []
        for i, symbol in enumerate(pair):
            if signal[i] == 0:
                continue

            sym_norm = symbol.replace("_", "")
            if open_positions.get(sym_norm, 0) >= self.max_position:
                self.logger.info(f"Max position reached for {symbol}, skipping.")
                continue

            base, quote = self._extract_symbol(symbol)
            price_i = self.get_imbalance_price(base, quote, 0.01)
            if price_i is None:
                self.logger.warning(f"Skipped {symbol}: No imbalance price available.")
                continue

            obi = self.get_imbalance_ob(base, quote, 0.01)
            if obi is None:
                self.logger.warning(f"Skipped {symbol}: No imbalance OB available.")
                continue

            # Order book imbalance filter
            if obi < -obi_threshold:
                obi_sig = -1
            elif obi > obi_threshold:
                obi_sig = 1
            else:
                obi_sig = 0

            # Order book imbalance filter (require confirmation)
            if obi_sig == 0 or obi_sig != signal[i]:
                order_results.append(f"{symbol} skipped: OB({obi_sig}) != Z({signal[i]})")
                continue

            # Limit price calculation with capped slippage
            slippage = np.clip(vola[i] * signal[i] * -1, -0.01, 0.01)
            lim_price = price_i * (1 + slippage)

            entry_dict[symbol] = (int(signal[i]), float(lim_price))
            self.logger.info(
                f"[ENTRY] {symbol} | Signal: {signal[i]:+d} | "
                f"Vola: {vola[i]:.4f} | Price: {price_i:.5f} "
                f"→ Limit: {lim_price:.5f} | OB: {obi:.4f} → Signal: {obi_sig:+d}"
            )

        return entry_dict

    def get_opened_positions(self) -> Dict[str, int]:
        """Return signed open position count per symbol."""
        open_pos = self.get_open_positions().get('data', [])
        if not open_pos:
            return {}

        all_symbols = [s.replace("_", "") for p in self.pairs for s in p]
        positions = {}

        for pos in open_pos:
            sym = pos['symbol']
            if sym not in all_symbols:
                continue

            value = abs(float(pos.get('positionValue', 0)))
            side = pos.get('side', '').upper()
            sign = 1 if side == 'BUY' else -1
            positions[sym] = positions.get(sym, 0) + int(round(sign * value / self.value_entry))

        return positions

    def calculate_position_size(self, value_entry: float = 10, price: Optional[float] = None) -> float:
        """Calculate position size based on available quote balance and risk pct."""
        balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0) * 10

        if self.value_entry > balance:
            self.logger.warning("Insufficient balance to open a new position.")
            return 0.0

        price = price or self.get_price(self.base, self.quote)
        if not price or price <= 0:
            self.logger.warning("Invalid price for sizing.")
            return 0.0
        
        quantity = round(value_entry / price, 6)
        return quantity
    
    def handle_orders(self):
        """Handle existing orders."""
        now = int(time.time() * 1000)
        expired_keys = []

        for ts, data in list(self.placed_orders.items()):
            if now - ts >= self.refresh_order * 1000:
                order_id = data.get('order_id')
                base = data.get('base')
                quote = data.get('quote')
                self.cancel_order(base=base, quote=quote, order_id=order_id)
                self.logger.info(f"Order {order_id} cancelled.")
                expired_keys.append(ts)

        for ts in expired_keys:
            del self.placed_orders[ts]
            
    def execute_trade(self, signal: int, price: float, quantity: float,
                      order_type: str = 'limit', force: str = 'GTC') -> bool:
        """
        Ensures delay between repeated executions using latest_entry_ts.
        """
        
        order_id = None
        if signal == 1:
            ask = self.get_price(self.base, self.quote, 'bid')
            price = min(ask, price)
            order_id = self.buy(quantity, price, order_type=order_type, force=force)

        elif signal == -1:
            bid = self.get_price(self.base, self.quote, 'ask')
            price = max(bid, price)
            order_id = self.sell(quantity, price, order_type=order_type, force=force)

        now = int(time.time() * 1000)
        if order_id:
            self.placed_orders[now] = {'order_id': order_id, 'base': self.base, 'quote': self.quote}

    def close_all_remaining_positions(self):
        """Close all remaining positions forcefully."""
        self.logger.info("Closing all remaining orders and positions forcefully...")
        self.position.cancel_all_orders(self, force_close=True)
        self.position.close_all_positions(self, force_close=True)

    def handle_stop_loss(self):
        """Check stop loss condition."""
        curr_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        if curr_balance < self.init_balance * (1 - self.stop_loss_threshold):
            self.logger.warning("Stop Loss triggered!")
            self.stop_flag = True
            self.handle_exit()

    def process_pair(self, pair):
        """Handles entry signals and order execution for a single pair."""
        try:
            signal = self.get_entry(pair)
            if not signal:
                return

            for symbol, (sig, lim_price) in signal.items():
                open_positions = self.get_opened_positions()
                if open_positions.get(symbol.replace("_", ""), 0) >= self.max_position:
                    self.logger.info(f"Max position reached for {symbol}, skipping entry.")
                    continue

                base, quote = self._extract_symbol(symbol)
                self.base, self.quote = base, quote

                price = self.get_price(self.base, self.quote)
                size = self.calculate_position_size(self.value_entry, price)
                if size > 0:
                    self.execute_trade(sig, lim_price, size, order_type='limit')
                else:
                    self.logger.warning("Zero position size, skipping trade.")

        except Exception as e:
            self.logger.error(f"Error processing pair {pair}: {e}")


    def run(self):
        """Run main strategy loop with concurrent execution per pair."""
        self.close_all_remaining_positions()
        self.logger.info("Starting Strategy...")
        self.init_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)

        try:
            while not self.stop_flag:
                self.handle_orders()
                self.handle_stop_loss()

                with ThreadPoolExecutor(max_workers=min(5, len(self.pairs))) as executor:
                    futures = [executor.submit(self.process_pair, pair) for pair in self.pairs]
                    for future in as_completed(futures):
                        future.result()  

                time.sleep(self.sleep_time)

            self.close_all_remaining_positions()
        except Exception as e:
            self.logger.exception(f"Run error: {e}")
        finally:
            self.handle_exit()
            self.logger.info("Strategy stopped cleanly.")

