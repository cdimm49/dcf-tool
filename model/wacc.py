def adjusted_beta(raw_beta):
    """Blume-adjusted beta: 2/3 x raw beta + 1/3 x 1.0.

    Betas measured from past returns drift toward the market average of 1.0 over
    time, so a raw beta of 0.15 overstates how little risk a stock carries. This
    is the adjustment Bloomberg reports as "adjusted beta".
    """
    return 2 / 3 * raw_beta + 1 / 3


def cost_of_equity(rf, beta, erp):
    """Calculate the CAPM cost of equity from the risk-free rate and ERP."""
    return rf + beta * erp


def wacc(
    equity_value,
    debt_value,
    cost_of_equity,
    pre_tax_cost_of_debt,
    tax_rate,
):
    """Calculate the debt-and-equity weighted average cost of capital."""
    total_capital = equity_value + debt_value
    if total_capital == 0:
        raise ValueError("Equity value and debt value cannot both sum to zero.")

    equity_weight = equity_value / total_capital
    debt_weight = debt_value / total_capital
    return (
        equity_weight * cost_of_equity
        + debt_weight * pre_tax_cost_of_debt * (1 - tax_rate)
    )
