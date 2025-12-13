"""
Visualization functions for strategy performance.
Uses matplotlib for chart generation.
"""

from pathlib import Path
from typing import Any

import numpy as np


def plot_equity_curve(
    equity_curve: list[float] | np.ndarray,
    dates: list[Any] | None = None,
    title: str = "Equity Curve",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 6)
) -> Any:
    """
    Plot portfolio equity curve.
    
    Args:
        equity_curve: Portfolio values over time
        dates: Optional date labels
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        ax.plot(dates, equity_curve, linewidth=1.5, color='#2E86AB')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
    else:
        ax.plot(equity_curve, linewidth=1.5, color='#2E86AB')
        ax.set_xlabel('Period')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    # Fill between
    ax.fill_between(
        range(len(equity_curve)) if dates is None else dates,
        equity_curve,
        equity_curve[0],
        alpha=0.2,
        color='#2E86AB'
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drawdown(
    equity_curve: list[float] | np.ndarray,
    dates: list[Any] | None = None,
    title: str = "Drawdown",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 4)
) -> Any:
    """
    Plot drawdown over time.
    
    Args:
        equity_curve: Portfolio values
        dates: Optional date labels
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100  # As percentage
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = dates if dates is not None else range(len(drawdown))
    ax.fill_between(x, 0, -drawdown, color='#E74C3C', alpha=0.6)
    ax.plot(x, -drawdown, color='#C0392B', linewidth=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Drawdown (%)')
    ax.set_ylim(top=0)
    ax.grid(True, alpha=0.3)
    
    # Mark max drawdown
    max_dd_idx = np.argmax(drawdown)
    ax.scatter([x[max_dd_idx]], [-drawdown[max_dd_idx]], 
               color='red', s=100, zorder=5, label=f'Max DD: {drawdown[max_dd_idx]:.1f}%')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_returns_distribution(
    returns: list[float] | np.ndarray,
    title: str = "Returns Distribution",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (10, 6)
) -> Any:
    """
    Plot histogram of returns with statistics.
    
    Args:
        returns: Array of returns
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from scipy import stats
    
    returns = np.array(returns)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram
    n, bins, patches = ax.hist(returns * 100, bins=50, density=True, 
                                alpha=0.7, color='#3498DB', edgecolor='white')
    
    # Fit normal distribution
    mu, std = returns.mean() * 100, returns.std() * 100
    x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal Fit')
    
    # Statistics box
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    stats_text = f'Mean: {mu:.2f}%\nStd: {std:.2f}%\nSkew: {skewness:.2f}\nKurtosis: {kurtosis:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Density')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trade_analysis(
    trades: list[dict[str, Any]],
    title: str = "Trade Analysis",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (14, 10)
) -> Any:
    """
    Plot comprehensive trade analysis.
    
    Args:
        trades: List of trade dicts
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    if not trades:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center')
        return fig
    
    pnls = [t.get('pnl', 0) for t in trades]
    cumulative_pnl = np.cumsum(pnls)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Cumulative P&L
    ax1 = axes[0, 0]
    ax1.plot(cumulative_pnl, linewidth=1.5, color='#2E86AB')
    ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0, 
                     where=[p >= 0 for p in cumulative_pnl], color='#27AE60', alpha=0.3)
    ax1.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=[p < 0 for p in cumulative_pnl], color='#E74C3C', alpha=0.3)
    ax1.set_title('Cumulative P&L')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('P&L ($)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual trade P&L
    ax2 = axes[0, 1]
    colors = ['#27AE60' if p > 0 else '#E74C3C' for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, edgecolor='white')
    ax2.set_title('Individual Trade P&L')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('P&L ($)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Win/Loss pie chart
    ax3 = axes[1, 0]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    breakeven = sum(1 for p in pnls if p == 0)
    
    sizes = [wins, losses, breakeven] if breakeven > 0 else [wins, losses]
    labels = ['Wins', 'Losses', 'Breakeven'] if breakeven > 0 else ['Wins', 'Losses']
    colors_pie = ['#27AE60', '#E74C3C', '#95A5A6'][:len(sizes)]
    
    ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=[0.05] * len(sizes))
    ax3.set_title(f'Win Rate: {wins}/{len(pnls)} ({wins/len(pnls)*100:.1f}%)')
    
    # 4. P&L distribution
    ax4 = axes[1, 1]
    ax4.hist(pnls, bins=30, color='#3498DB', alpha=0.7, edgecolor='white')
    ax4.axvline(x=np.mean(pnls), color='red', linestyle='--', label=f'Mean: ${np.mean(pnls):.2f}')
    ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_title('P&L Distribution')
    ax4.set_xlabel('P&L ($)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_monthly_returns_heatmap(
    monthly_returns: dict[str, float],
    title: str = "Monthly Returns Heatmap",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 6)
) -> Any:
    """
    Plot monthly returns as a heatmap.
    
    Args:
        monthly_returns: Dict mapping "YYYY-MM" to return
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    
    if not monthly_returns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No monthly data', ha='center', va='center')
        return fig
    
    # Parse into year/month matrix
    years = sorted(set(k[:4] for k in monthly_returns.keys()))
    months = list(range(1, 13))
    
    data = np.full((len(years), 12), np.nan)
    
    for key, value in monthly_returns.items():
        year_idx = years.index(key[:4])
        month_idx = int(key[5:7]) - 1
        data[year_idx, month_idx] = value * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap with center at 0
    vmax = np.nanmax(np.abs(data))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', norm=norm)
    
    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    
    # Annotate cells
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(data[i, j]):
                color = 'white' if abs(data[i, j]) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center', 
                       fontsize=9, color=color)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Return (%)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rolling_metrics(
    equity_curve: list[float] | np.ndarray,
    window: int = 30,
    title: str = "Rolling Performance Metrics",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (12, 10)
) -> Any:
    """
    Plot rolling Sharpe, volatility, and returns.
    
    Args:
        equity_curve: Portfolio values
        window: Rolling window size
        title: Chart title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from reports.metrics import calculate_returns
    
    equity = np.array(equity_curve)
    returns = calculate_returns(equity)
    
    if len(returns) < window:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'Need at least {window} periods', ha='center', va='center')
        return fig
    
    # Calculate rolling metrics
    rolling_returns = []
    rolling_volatility = []
    rolling_sharpe = []
    
    for i in range(window, len(returns) + 1):
        window_returns = returns[i-window:i]
        rolling_returns.append(np.mean(window_returns) * 252 * 100)  # Annualized %
        rolling_volatility.append(np.std(window_returns) * np.sqrt(252) * 100)
        sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252) if np.std(window_returns) > 0 else 0
        rolling_sharpe.append(sharpe)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    x = range(window, len(returns) + 1)
    
    # Rolling Return
    ax1 = axes[0]
    ax1.plot(x, rolling_returns, color='#2E86AB', linewidth=1.5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Ann. Return (%)')
    ax1.set_title(f'{window}-Period Rolling Annualized Return')
    ax1.grid(True, alpha=0.3)
    
    # Rolling Volatility
    ax2 = axes[1]
    ax2.plot(x, rolling_volatility, color='#E74C3C', linewidth=1.5)
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title(f'{window}-Period Rolling Volatility')
    ax2.grid(True, alpha=0.3)
    
    # Rolling Sharpe
    ax3 = axes[2]
    ax3.plot(x, rolling_sharpe, color='#27AE60', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_xlabel('Period')
    ax3.set_title(f'{window}-Period Rolling Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_performance_dashboard(
    equity_curve: list[float] | np.ndarray,
    trades: list[dict[str, Any]],
    dates: list[Any] | None = None,
    title: str = "Performance Dashboard",
    save_path: Path | None = None,
    figsize: tuple[int, int] = (16, 12)
) -> Any:
    """
    Create comprehensive performance dashboard.
    
    Args:
        equity_curve: Portfolio values
        trades: List of trade dicts
        dates: Optional date labels
        title: Dashboard title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from reports.metrics import (
        calculate_returns, calculate_max_drawdown, 
        compute_strategy_metrics
    )
    
    equity = np.array(equity_curve)
    returns = calculate_returns(equity)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Equity Curve (top, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    x = dates if dates is not None else range(len(equity))
    ax1.plot(x, equity, linewidth=1.5, color='#2E86AB')
    ax1.fill_between(x, equity, equity[0], alpha=0.2, color='#2E86AB')
    ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Key Metrics Box (top right)
    ax_metrics = fig.add_subplot(gs[0, 2])
    ax_metrics.axis('off')
    
    metrics = compute_strategy_metrics(equity_curve, trades)
    metrics_text = f"""
Key Metrics
═══════════════
Total Return: {metrics.total_return:.2%}
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Sortino Ratio: {metrics.sortino_ratio:.2f}
Max Drawdown: {metrics.max_drawdown:.2%}

Total Trades: {metrics.total_trades}
Win Rate: {metrics.win_rate:.2%}
Profit Factor: {metrics.profit_factor:.2f}
Expectancy: ${metrics.expectancy:.2f}
"""
    ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.8))
    
    # 3. Drawdown (middle left)
    ax2 = fig.add_subplot(gs[1, 0])
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    ax2.fill_between(range(len(drawdown)), 0, -drawdown, color='#E74C3C', alpha=0.6)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 4. Returns Distribution (middle center)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(returns * 100, bins=30, color='#3498DB', alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=np.mean(returns) * 100, color='red', linestyle='-', 
                label=f'Mean: {np.mean(returns)*100:.2f}%')
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Return (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 5. Win/Loss (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    pnls = [t.get('pnl', 0) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    ax4.pie([wins, losses], labels=['Wins', 'Losses'], colors=['#27AE60', '#E74C3C'],
            autopct='%1.1f%%', startangle=90, explode=[0.05, 0.05])
    ax4.set_title(f'Win Rate ({wins}/{len(trades)})', fontsize=12, fontweight='bold')
    
    # 6. Cumulative P&L (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    cumulative_pnl = np.cumsum(pnls)
    ax5.plot(cumulative_pnl, linewidth=1.5, color='#2E86AB')
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Trade #')
    ax5.set_ylabel('P&L ($)')
    ax5.grid(True, alpha=0.3)
    
    # 7. Trade P&L bars (bottom center)
    ax6 = fig.add_subplot(gs[2, 1])
    colors = ['#27AE60' if p > 0 else '#E74C3C' for p in pnls]
    ax6.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_title('Individual Trade P&L', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Trade #')
    ax6.grid(True, alpha=0.3)
    
    # 8. Trade Size Distribution (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    sizes = [abs(t.get('size_usd', 0)) for t in trades if t.get('size_usd')]
    if sizes:
        ax7.hist(sizes, bins=20, color='#9B59B6', alpha=0.7, edgecolor='white')
        ax7.set_title('Position Size Distribution', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Size ($)')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No size data', ha='center', va='center')
        ax7.set_title('Position Sizes', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_html_report(
    equity_curve: list[float] | np.ndarray,
    trades: list[dict[str, Any]],
    output_path: Path,
    title: str = "Strategy Performance Report"
) -> None:
    """
    Generate HTML performance report.
    
    Args:
        equity_curve: Portfolio values
        trades: List of trades
        output_path: Output HTML path
        title: Report title
    """
    import base64
    from io import BytesIO
    from reports.metrics import compute_strategy_metrics
    
    # Generate plots
    fig_equity = plot_equity_curve(equity_curve, title="Equity Curve")
    fig_trades = plot_trade_analysis(trades, title="Trade Analysis")
    
    # Convert to base64
    def fig_to_base64(fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    equity_img = fig_to_base64(fig_equity)
    trades_img = fig_to_base64(fig_trades)
    
    import matplotlib.pyplot as plt
    plt.close(fig_equity)
    plt.close(fig_trades)
    
    # Compute metrics
    metrics = compute_strategy_metrics(equity_curve, trades)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .chart {{ margin: 30px 0; text-align: center; }}
        .chart img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.total_return >= 0 else 'negative'}">{metrics.total_return:.2%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{metrics.max_drawdown:.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${metrics.expectancy:.2f}</div>
                <div class="metric-label">Expectancy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.total_pnl >= 0 else 'negative'}">${metrics.total_pnl:,.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
        </div>
        
        <h2>Equity Curve</h2>
        <div class="chart">
            <img src="data:image/png;base64,{equity_img}" alt="Equity Curve">
        </div>
        
        <h2>Trade Analysis</h2>
        <div class="chart">
            <img src="data:image/png;base64,{trades_img}" alt="Trade Analysis">
        </div>
        
        <h2>Detailed Metrics</h2>
        <pre>{metrics.summary()}</pre>
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
