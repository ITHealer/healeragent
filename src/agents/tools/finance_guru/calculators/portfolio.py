"""
Finance Guru - Portfolio Calculators (Phase 4)

Layer 2: Pure computation functions for portfolio optimization and correlation analysis.

This module implements:
- Portfolio optimization (Mean-Variance, Min Variance, Max Sharpe, Risk Parity)
- Correlation and covariance matrix calculations
- Efficient frontier generation
- Rebalancing suggestions

MATHEMATICAL FOUNDATION:
- Portfolio Return: R_p = Σ(w_i * R_i)
- Portfolio Variance: σ²_p = Σ Σ(w_i * w_j * σ_ij)
- Sharpe Ratio: (R_p - R_f) / σ_p
- Correlation: ρ_ij = σ_ij / (σ_i * σ_j)

Author: HealerAgent Development Team
"""

import math
from typing import Optional
import numpy as np
from scipy import optimize
from scipy import stats

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationContext,
    CalculationError,
    InsufficientDataError,
)
from src.agents.tools.finance_guru.models.portfolio import (
    # Enums
    OptimizationMethod,
    CorrelationMethod,
    RiskModel,
    # Input
    PortfolioDataInput,
    PortfolioPriceData,
    OptimizationConfig,
    CorrelationConfig,
    EfficientFrontierConfig,
    # Output
    AssetAllocation,
    OptimizationOutput,
    FrontierPoint,
    EfficientFrontierOutput,
    CorrelationMatrixOutput,
    CovarianceMatrixOutput,
    RollingCorrelationOutput,
    PortfolioCorrelationOutput,
    RebalancingSuggestion,
    RebalancingOutput,
)


class PortfolioOptimizer(BaseCalculator):
    """Portfolio optimization calculator.

    Implements multiple optimization methods:
    - Mean-Variance: Classic Markowitz optimization
    - Min Variance: Minimize portfolio volatility
    - Max Sharpe: Maximize risk-adjusted returns
    - Risk Parity: Equal risk contribution from each asset
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="PortfolioOptimizer")

    def calculate(self, data, **kwargs):
        """Implement abstract method - routes to optimize()."""
        config = kwargs.get("config", OptimizationConfig())
        return self.optimize(data, config)

    def _calculate_returns(self, prices: list[float]) -> np.ndarray:
        """Calculate returns from price series."""
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        return returns

    def _get_returns_matrix(self, data: PortfolioDataInput) -> np.ndarray:
        """Get returns matrix for all assets."""
        returns_list = []
        for asset in data.assets:
            if asset.returns:
                returns_list.append(np.array(asset.returns))
            else:
                returns_list.append(self._calculate_returns(asset.prices))
        return np.array(returns_list)

    def _portfolio_return(self, weights: np.ndarray, mean_returns: np.ndarray) -> float:
        """Calculate portfolio expected return."""
        return float(np.dot(weights, mean_returns))

    def _portfolio_volatility(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return float(np.sqrt(variance))

    def _portfolio_sharpe(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float,
    ) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self._portfolio_return(weights, mean_returns)
        vol = self._portfolio_volatility(weights, cov_matrix)
        if vol == 0:
            return 0.0
        return float((ret - risk_free_rate) / vol)

    def _risk_contribution(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each asset."""
        port_vol = self._portfolio_volatility(weights, cov_matrix)
        if port_vol == 0:
            return np.zeros(len(weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / port_vol
        return risk_contrib

    def optimize(
        self,
        data: PortfolioDataInput,
        config: OptimizationConfig,
    ) -> OptimizationOutput:
        """Run portfolio optimization.

        Args:
            data: Portfolio data with asset prices/returns
            config: Optimization configuration

        Returns:
            OptimizationOutput with optimal allocations
        """
        self.context.start()

        try:
            # Get returns matrix and statistics
            returns_matrix = self._get_returns_matrix(data)
            n_assets = len(data.assets)

            # Annualize statistics
            mean_returns = np.mean(returns_matrix, axis=1) * data.trading_days_per_year
            cov_matrix = np.cov(returns_matrix) * data.trading_days_per_year

            # Ensure covariance is 2D even for 2 assets
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            elif cov_matrix.ndim == 1:
                cov_matrix = np.diag(cov_matrix)

            symbols = [asset.symbol for asset in data.assets]
            risk_free_rate = data.risk_free_rate

            # Select optimization method
            if config.method == OptimizationMethod.MIN_VARIANCE:
                weights = self._optimize_min_variance(n_assets, cov_matrix, config)
            elif config.method == OptimizationMethod.MAX_SHARPE:
                weights = self._optimize_max_sharpe(
                    n_assets, mean_returns, cov_matrix, risk_free_rate, config
                )
            elif config.method == OptimizationMethod.RISK_PARITY:
                weights = self._optimize_risk_parity(n_assets, cov_matrix, config)
            else:  # Mean-Variance with target return
                weights = self._optimize_mean_variance(
                    n_assets, mean_returns, cov_matrix, config
                )

            # Calculate portfolio metrics
            port_return = self._portfolio_return(weights, mean_returns)
            port_vol = self._portfolio_volatility(weights, cov_matrix)
            sharpe = self._portfolio_sharpe(weights, mean_returns, cov_matrix, risk_free_rate)
            risk_contribs = self._risk_contribution(weights, cov_matrix)

            # Build allocations
            allocations = [
                AssetAllocation(
                    symbol=symbols[i],
                    weight=float(weights[i]),
                    expected_return=float(mean_returns[i]),
                    risk_contribution=float(risk_contribs[i]) if i < len(risk_contribs) else 0.0,
                )
                for i in range(n_assets)
            ]

            self.context.complete()

            return OptimizationOutput(
                allocations=allocations,
                expected_return=port_return,
                expected_volatility=port_vol,
                sharpe_ratio=sharpe,
                method_used=config.method,
                converged=True,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Optimization failed: {e}")

    def _optimize_min_variance(
        self,
        n_assets: int,
        cov_matrix: np.ndarray,
        config: OptimizationConfig,
    ) -> np.ndarray:
        """Find minimum variance portfolio."""
        # Objective: minimize portfolio variance
        def objective(w):
            return self._portfolio_volatility(w, cov_matrix)

        # Constraints: weights sum to 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # Bounds: min and max weights
        bounds = tuple((config.min_weight, config.max_weight) for _ in range(n_assets))

        # Initial guess: equal weights
        initial = np.array([1.0 / n_assets] * n_assets)

        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _optimize_max_sharpe(
        self,
        n_assets: int,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float,
        config: OptimizationConfig,
    ) -> np.ndarray:
        """Find maximum Sharpe ratio portfolio."""
        # Objective: minimize negative Sharpe
        def objective(w):
            return -self._portfolio_sharpe(w, mean_returns, cov_matrix, risk_free_rate)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((config.min_weight, config.max_weight) for _ in range(n_assets))
        initial = np.array([1.0 / n_assets] * n_assets)

        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _optimize_mean_variance(
        self,
        n_assets: int,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        config: OptimizationConfig,
    ) -> np.ndarray:
        """Mean-variance optimization with target return."""
        target_return = config.target_return or float(np.mean(mean_returns))

        def objective(w):
            return self._portfolio_volatility(w, cov_matrix)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: self._portfolio_return(w, mean_returns) - target_return},
        ]

        bounds = tuple((config.min_weight, config.max_weight) for _ in range(n_assets))
        initial = np.array([1.0 / n_assets] * n_assets)

        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def _optimize_risk_parity(
        self,
        n_assets: int,
        cov_matrix: np.ndarray,
        config: OptimizationConfig,
    ) -> np.ndarray:
        """Risk parity: equal risk contribution from each asset."""
        target_risk = 1.0 / n_assets

        def objective(w):
            risk_contribs = self._risk_contribution(w, cov_matrix)
            # Minimize squared deviation from target risk
            return np.sum((risk_contribs - target_risk) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((max(0.01, config.min_weight), config.max_weight) for _ in range(n_assets))
        initial = np.array([1.0 / n_assets] * n_assets)

        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def generate_efficient_frontier(
        self,
        data: PortfolioDataInput,
        config: EfficientFrontierConfig,
    ) -> EfficientFrontierOutput:
        """Generate efficient frontier.

        Args:
            data: Portfolio data
            config: Frontier configuration

        Returns:
            EfficientFrontierOutput with frontier points
        """
        self.context.start()

        try:
            returns_matrix = self._get_returns_matrix(data)
            n_assets = len(data.assets)

            mean_returns = np.mean(returns_matrix, axis=1) * data.trading_days_per_year
            cov_matrix = np.cov(returns_matrix) * data.trading_days_per_year

            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            elif cov_matrix.ndim == 1:
                cov_matrix = np.diag(cov_matrix)

            symbols = [asset.symbol for asset in data.assets]
            risk_free_rate = data.risk_free_rate

            # Determine return range
            if config.return_range:
                min_ret, max_ret = config.return_range
            else:
                min_ret = float(np.min(mean_returns))
                max_ret = float(np.max(mean_returns))

            target_returns = np.linspace(min_ret, max_ret, config.num_portfolios)

            frontier_points = []
            opt_config = OptimizationConfig(method=OptimizationMethod.MEAN_VARIANCE, min_weight=0.0)

            for target in target_returns:
                opt_config.target_return = float(target)
                try:
                    weights = self._optimize_mean_variance(n_assets, mean_returns, cov_matrix, opt_config)
                    port_ret = self._portfolio_return(weights, mean_returns)
                    port_vol = self._portfolio_volatility(weights, cov_matrix)
                    sharpe = self._portfolio_sharpe(weights, mean_returns, cov_matrix, risk_free_rate)

                    frontier_points.append(FrontierPoint(
                        expected_return=port_ret,
                        expected_volatility=port_vol,
                        sharpe_ratio=sharpe,
                        weights={symbols[i]: float(weights[i]) for i in range(n_assets)},
                    ))
                except Exception:
                    continue

            # Find max Sharpe and min variance portfolios
            max_sharpe_config = OptimizationConfig(method=OptimizationMethod.MAX_SHARPE)
            max_sharpe_weights = self._optimize_max_sharpe(
                n_assets, mean_returns, cov_matrix, risk_free_rate, max_sharpe_config
            )
            max_sharpe_point = FrontierPoint(
                expected_return=self._portfolio_return(max_sharpe_weights, mean_returns),
                expected_volatility=self._portfolio_volatility(max_sharpe_weights, cov_matrix),
                sharpe_ratio=self._portfolio_sharpe(
                    max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate
                ),
                weights={symbols[i]: float(max_sharpe_weights[i]) for i in range(n_assets)},
            )

            min_var_config = OptimizationConfig(method=OptimizationMethod.MIN_VARIANCE)
            min_var_weights = self._optimize_min_variance(n_assets, cov_matrix, min_var_config)
            min_var_point = FrontierPoint(
                expected_return=self._portfolio_return(min_var_weights, mean_returns),
                expected_volatility=self._portfolio_volatility(min_var_weights, cov_matrix),
                sharpe_ratio=self._portfolio_sharpe(
                    min_var_weights, mean_returns, cov_matrix, risk_free_rate
                ),
                weights={symbols[i]: float(min_var_weights[i]) for i in range(n_assets)},
            )

            # Individual asset points
            asset_points = None
            if config.include_assets:
                asset_points = {
                    symbols[i]: (
                        float(np.sqrt(cov_matrix[i, i])),
                        float(mean_returns[i]),
                    )
                    for i in range(n_assets)
                }

            self.context.complete()

            return EfficientFrontierOutput(
                frontier_points=frontier_points,
                max_sharpe_portfolio=max_sharpe_point,
                min_variance_portfolio=min_var_point,
                asset_points=asset_points,
                symbols=symbols,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Efficient frontier calculation failed: {e}")


class CorrelationEngine(BaseCalculator):
    """Correlation and covariance analysis calculator.

    Computes:
    - Correlation matrices (Pearson, Spearman, Kendall)
    - Covariance matrices (Sample, Shrinkage, Exponential)
    - Rolling correlations
    - Portfolio diversification metrics
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="CorrelationEngine")

    def calculate(self, data, **kwargs):
        """Implement abstract method - routes to calculate_correlation_matrix()."""
        config = kwargs.get("config", CorrelationConfig())
        return self.calculate_correlation_matrix(data, config)

    def _calculate_returns_from_prices(self, prices: list[float]) -> np.ndarray:
        """Calculate returns from prices."""
        arr = np.array(prices)
        return np.diff(arr) / arr[:-1]

    def _prices_to_returns_matrix(self, data: PortfolioPriceData) -> np.ndarray:
        """Convert price matrix to returns matrix."""
        returns = []
        for row in data.price_matrix:
            returns.append(self._calculate_returns_from_prices(row))
        return np.array(returns)

    def calculate_correlation_matrix(
        self,
        data: PortfolioPriceData,
        config: CorrelationConfig,
    ) -> CorrelationMatrixOutput:
        """Calculate correlation matrix.

        Args:
            data: Price data for multiple assets
            config: Correlation configuration

        Returns:
            CorrelationMatrixOutput with matrix and statistics
        """
        self.context.start()

        try:
            returns_matrix = self._prices_to_returns_matrix(data)
            n_assets = len(data.symbols)

            if returns_matrix.shape[1] < config.min_periods:
                raise InsufficientDataError(
                    f"Need at least {config.min_periods} periods, got {returns_matrix.shape[1]}"
                )

            # Calculate correlation based on method
            if config.method == CorrelationMethod.PEARSON:
                corr_matrix = np.corrcoef(returns_matrix)
            elif config.method == CorrelationMethod.SPEARMAN:
                corr_matrix, _ = stats.spearmanr(returns_matrix.T)
                if n_assets == 2:
                    corr_matrix = np.array([[1, corr_matrix], [corr_matrix, 1]])
            else:  # Kendall
                corr_matrix = np.zeros((n_assets, n_assets))
                for i in range(n_assets):
                    for j in range(n_assets):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                        elif i < j:
                            tau, _ = stats.kendalltau(returns_matrix[i], returns_matrix[j])
                            corr_matrix[i, j] = tau
                            corr_matrix[j, i] = tau

            # Convert to nested dict
            matrix_dict = {
                data.symbols[i]: {
                    data.symbols[j]: float(corr_matrix[i, j])
                    for j in range(n_assets)
                }
                for i in range(n_assets)
            }

            # Find min/max correlations (excluding diagonal)
            min_corr = (data.symbols[0], data.symbols[1], 1.0)
            max_corr = (data.symbols[0], data.symbols[1], -1.0)
            correlations = []

            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    c = corr_matrix[i, j]
                    correlations.append(c)
                    if c < min_corr[2]:
                        min_corr = (data.symbols[i], data.symbols[j], float(c))
                    if c > max_corr[2]:
                        max_corr = (data.symbols[i], data.symbols[j], float(c))

            avg_corr = float(np.mean(correlations)) if correlations else 0.0

            self.context.complete()

            return CorrelationMatrixOutput(
                matrix=matrix_dict,
                symbols=data.symbols,
                method=config.method,
                average_correlation=avg_corr,
                min_correlation=min_corr,
                max_correlation=max_corr,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except InsufficientDataError:
            raise
        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Correlation calculation failed: {e}")

    def calculate_covariance_matrix(
        self,
        data: PortfolioPriceData,
        config: CorrelationConfig,
    ) -> CovarianceMatrixOutput:
        """Calculate covariance matrix.

        Args:
            data: Price data for multiple assets
            config: Configuration including risk model

        Returns:
            CovarianceMatrixOutput with matrix and statistics
        """
        self.context.start()

        try:
            returns_matrix = self._prices_to_returns_matrix(data)
            n_assets = len(data.symbols)

            if returns_matrix.shape[1] < config.min_periods:
                raise InsufficientDataError(
                    f"Need at least {config.min_periods} periods, got {returns_matrix.shape[1]}"
                )

            # Calculate covariance based on risk model
            if config.risk_model == RiskModel.SAMPLE:
                cov_matrix = np.cov(returns_matrix)
            elif config.risk_model == RiskModel.SHRINKAGE:
                cov_matrix = self._ledoit_wolf_shrinkage(returns_matrix)
            else:  # Exponential
                cov_matrix = self._exponential_covariance(returns_matrix)

            # Ensure 2D
            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            elif cov_matrix.ndim == 1:
                cov_matrix = np.diag(cov_matrix)

            # Annualize if requested
            if config.annualize:
                cov_matrix = cov_matrix * config.trading_days

            # Convert to nested dict
            matrix_dict = {
                data.symbols[i]: {
                    data.symbols[j]: float(cov_matrix[i, j])
                    for j in range(n_assets)
                }
                for i in range(n_assets)
            }

            # Calculate determinant and condition number
            try:
                determinant = float(np.linalg.det(cov_matrix))
                cond_number = float(np.linalg.cond(cov_matrix))
            except Exception:
                determinant = None
                cond_number = None

            self.context.complete()

            return CovarianceMatrixOutput(
                matrix=matrix_dict,
                symbols=data.symbols,
                risk_model=config.risk_model,
                is_annualized=config.annualize,
                determinant=determinant,
                condition_number=cond_number,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except InsufficientDataError:
            raise
        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Covariance calculation failed: {e}")

    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf shrinkage estimator for covariance."""
        n_assets, n_obs = returns.shape
        sample_cov = np.cov(returns)

        if sample_cov.ndim == 0:
            return np.array([[sample_cov]])

        # Shrinkage target: scaled identity
        mu = np.trace(sample_cov) / n_assets
        target = mu * np.eye(n_assets)

        # Calculate shrinkage intensity (simplified)
        delta = sample_cov - target
        shrinkage_intensity = min(1.0, max(0.0, 1.0 / n_obs))

        # Shrunk covariance
        return (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target

    def _exponential_covariance(
        self, returns: np.ndarray, halflife: int = 60
    ) -> np.ndarray:
        """Exponentially weighted covariance matrix."""
        n_assets, n_obs = returns.shape

        # Calculate exponential weights
        decay = np.log(2) / halflife
        weights = np.exp(-decay * np.arange(n_obs)[::-1])
        weights = weights / weights.sum()

        # Weighted mean
        weighted_mean = np.average(returns, axis=1, weights=weights)

        # Centered returns
        centered = returns - weighted_mean[:, np.newaxis]

        # Weighted covariance
        cov_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(i, n_assets):
                cov_ij = np.sum(weights * centered[i] * centered[j])
                cov_matrix[i, j] = cov_ij
                cov_matrix[j, i] = cov_ij

        return cov_matrix

    def calculate_rolling_correlation(
        self,
        data: PortfolioPriceData,
        asset1_idx: int,
        asset2_idx: int,
        config: CorrelationConfig,
    ) -> RollingCorrelationOutput:
        """Calculate rolling correlation between two assets.

        Args:
            data: Price data
            asset1_idx: Index of first asset
            asset2_idx: Index of second asset
            config: Configuration with rolling window

        Returns:
            RollingCorrelationOutput with time series
        """
        self.context.start()

        if config.rolling_window is None:
            raise CalculationError("rolling_window must be specified for rolling correlation")

        try:
            returns_matrix = self._prices_to_returns_matrix(data)
            window = config.rolling_window
            n_obs = returns_matrix.shape[1]

            if n_obs < window:
                raise InsufficientDataError(
                    f"Need at least {window} observations, got {n_obs}"
                )

            returns1 = returns_matrix[asset1_idx]
            returns2 = returns_matrix[asset2_idx]

            # Calculate rolling correlation
            correlations = []
            for i in range(window, n_obs + 1):
                r1_window = returns1[i - window:i]
                r2_window = returns2[i - window:i]
                corr = np.corrcoef(r1_window, r2_window)[0, 1]
                correlations.append(float(corr))

            # Determine trend
            avg_corr = float(np.mean(correlations))
            std_corr = float(np.std(correlations))
            current = correlations[-1]

            # Simple trend detection
            first_half = np.mean(correlations[:len(correlations) // 2])
            second_half = np.mean(correlations[len(correlations) // 2:])

            if second_half - first_half > 0.1:
                trend = "up"
            elif first_half - second_half > 0.1:
                trend = "down"
            else:
                trend = "stable"

            # Get dates if available
            dates = None
            if data.dates and len(data.dates) >= len(correlations):
                dates = data.dates[window:]

            self.context.complete()

            return RollingCorrelationOutput(
                asset_pair=(data.symbols[asset1_idx], data.symbols[asset2_idx]),
                correlations=correlations,
                dates=dates,
                window_size=window,
                average=avg_corr,
                std=std_corr,
                trend=trend,
                current=current,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except InsufficientDataError:
            raise
        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Rolling correlation failed: {e}")

    def analyze_portfolio_correlation(
        self,
        data: PortfolioPriceData,
        weights: list[float],
        config: CorrelationConfig,
    ) -> PortfolioCorrelationOutput:
        """Complete portfolio correlation analysis.

        Args:
            data: Price data for assets
            weights: Current portfolio weights
            config: Analysis configuration

        Returns:
            PortfolioCorrelationOutput with comprehensive analysis
        """
        self.context.start()

        try:
            # Calculate correlation and covariance matrices
            corr_output = self.calculate_correlation_matrix(data, config)
            cov_output = self.calculate_covariance_matrix(data, config)

            n_assets = len(data.symbols)
            weights_arr = np.array(weights)

            # Get covariance matrix as array
            cov_matrix = np.array([
                [cov_output.matrix[data.symbols[i]][data.symbols[j]]
                 for j in range(n_assets)]
                for i in range(n_assets)
            ])

            # Calculate diversification ratio
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights_arr, individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights_arr, np.dot(cov_matrix, weights_arr)))

            if portfolio_vol > 0:
                diversification_ratio = weighted_avg_vol / portfolio_vol
            else:
                diversification_ratio = 1.0

            # Calculate effective N (effective number of independent assets)
            corr_matrix = np.array([
                [corr_output.matrix[data.symbols[i]][data.symbols[j]]
                 for j in range(n_assets)]
                for i in range(n_assets)
            ])
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            if np.sum(eigenvalues) > 0:
                effective_n = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
            else:
                effective_n = n_assets

            # Get highest and lowest correlations
            correlations = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    c = corr_output.matrix[data.symbols[i]][data.symbols[j]]
                    correlations.append((data.symbols[i], data.symbols[j], c))

            correlations.sort(key=lambda x: x[2])
            lowest = correlations[:3] if len(correlations) >= 3 else correlations
            highest = correlations[-3:][::-1] if len(correlations) >= 3 else correlations[::-1]

            # Generate recommendations
            recommendations = []
            if diversification_ratio < 1.1:
                recommendations.append(
                    "Portfolio is not well diversified. Consider adding uncorrelated assets."
                )
            if effective_n < n_assets * 0.5:
                recommendations.append(
                    f"Effective diversification is only {effective_n:.1f} assets out of {n_assets}."
                )
            if highest and highest[0][2] > 0.8:
                recommendations.append(
                    f"High correlation ({highest[0][2]:.2f}) between {highest[0][0]} and {highest[0][1]}. "
                    "Consider reducing exposure to one."
                )

            self.context.complete()

            return PortfolioCorrelationOutput(
                correlation_matrix=corr_output,
                covariance_matrix=cov_output,
                diversification_ratio=float(diversification_ratio),
                effective_n=float(effective_n),
                highest_correlations=highest,
                lowest_correlations=lowest,
                recommendations=recommendations,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Portfolio correlation analysis failed: {e}")


class RebalancingCalculator(BaseCalculator):
    """Calculator for portfolio rebalancing suggestions."""

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="RebalancingCalculator")
        self.optimizer = PortfolioOptimizer()

    def calculate(self, data, **kwargs):
        """Implement abstract method - routes to suggest_rebalancing()."""
        config = kwargs.get("config", OptimizationConfig())
        return self.suggest_rebalancing(data, config)

    def suggest_rebalancing(
        self,
        data: PortfolioDataInput,
        config: OptimizationConfig,
        threshold: float = 0.05,
        transaction_cost: float = 0.001,
    ) -> RebalancingOutput:
        """Generate rebalancing suggestions.

        Args:
            data: Portfolio data with current weights
            config: Optimization configuration for target
            threshold: Minimum weight deviation to trigger rebalance
            transaction_cost: Cost per unit traded (e.g., 0.001 = 0.1%)

        Returns:
            RebalancingOutput with suggestions
        """
        self.context.start()

        try:
            # Get optimal portfolio
            optimal = self.optimizer.optimize(data, config)

            # Calculate returns for Sharpe calculation
            returns_matrix = self.optimizer._get_returns_matrix(data)
            mean_returns = np.mean(returns_matrix, axis=1) * data.trading_days_per_year
            cov_matrix = np.cov(returns_matrix) * data.trading_days_per_year

            if cov_matrix.ndim == 0:
                cov_matrix = np.array([[cov_matrix]])
            elif cov_matrix.ndim == 1:
                cov_matrix = np.diag(cov_matrix)

            # Current weights and Sharpe
            current_weights = np.array([asset.weight for asset in data.assets])
            current_sharpe = self.optimizer._portfolio_sharpe(
                current_weights, mean_returns, cov_matrix, data.risk_free_rate
            )

            suggestions = []
            total_turnover = 0.0

            for i, asset in enumerate(data.assets):
                target_weight = optimal.allocations[i].weight
                current_weight = asset.weight
                diff = target_weight - current_weight

                if abs(diff) >= threshold:
                    if diff > 0:
                        action = "buy"
                    elif diff < 0:
                        action = "sell"
                    else:
                        action = "hold"

                    if abs(diff) >= 0.15:
                        urgency = "high"
                    elif abs(diff) >= 0.08:
                        urgency = "medium"
                    else:
                        urgency = "low"

                    reason = (
                        f"Weight deviation of {abs(diff):.1%} from optimal. "
                        f"Current: {current_weight:.1%}, Target: {target_weight:.1%}"
                    )

                    suggestions.append(RebalancingSuggestion(
                        symbol=asset.symbol,
                        current_weight=current_weight,
                        target_weight=target_weight,
                        action=action,
                        urgency=urgency,
                        reason=reason,
                    ))

                    total_turnover += abs(diff)

            # Estimate transaction cost
            estimated_cost = total_turnover * transaction_cost

            # Calculate improvement
            improvement = {
                "sharpe_improvement": optimal.sharpe_ratio - current_sharpe,
                "volatility_reduction": (
                    self.optimizer._portfolio_volatility(current_weights, cov_matrix)
                    - optimal.expected_volatility
                ),
            }

            self.context.complete()

            return RebalancingOutput(
                suggestions=suggestions,
                total_turnover=total_turnover,
                estimated_cost=estimated_cost,
                current_sharpe=current_sharpe,
                target_sharpe=optimal.sharpe_ratio,
                improvement=improvement,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Rebalancing calculation failed: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def optimize_portfolio(
    data: PortfolioDataInput,
    method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
    **kwargs,
) -> OptimizationOutput:
    """Convenience function for portfolio optimization.

    Args:
        data: Portfolio data
        method: Optimization method
        **kwargs: Additional OptimizationConfig parameters

    Returns:
        OptimizationOutput
    """
    config = OptimizationConfig(method=method, **kwargs)
    optimizer = PortfolioOptimizer()
    return optimizer.optimize(data, config)


def calculate_correlation(
    data: PortfolioPriceData,
    method: CorrelationMethod = CorrelationMethod.PEARSON,
) -> CorrelationMatrixOutput:
    """Convenience function for correlation calculation.

    Args:
        data: Price data
        method: Correlation method

    Returns:
        CorrelationMatrixOutput
    """
    config = CorrelationConfig(method=method)
    engine = CorrelationEngine()
    return engine.calculate_correlation_matrix(data, config)


def generate_frontier(
    data: PortfolioDataInput,
    num_portfolios: int = 50,
) -> EfficientFrontierOutput:
    """Convenience function for efficient frontier.

    Args:
        data: Portfolio data
        num_portfolios: Number of frontier points

    Returns:
        EfficientFrontierOutput
    """
    config = EfficientFrontierConfig(num_portfolios=num_portfolios)
    optimizer = PortfolioOptimizer()
    return optimizer.generate_efficient_frontier(data, config)
