# stajax: Causal Quantitative Trading with JAX

[![PyPI version](https://badge.fury.io/py/stajax.svg)](https://badge.fury.io/py/stajax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

`stajax` is a high-performance Python library for causal quantitative trading, leveraging the power of JAX for GPU-accelerated computations. It combines causal inference, graph neural networks, and adaptive learning techniques to provide cutting-edge tools for financial modeling and strategy development.

## Features

- **Causal Graph Neural Networks (CGNN)**: Model complex market relationships and discover causal patterns.
- **Adaptive Frequency Domain Analysis**: Real-time signal processing for market data.
- **Quantum-Inspired Optimization**: Advanced portfolio optimization techniques.
- **Evolutionary Algorithms**: Develop and refine trading strategies automatically.
- **Reinforcement Learning**: Adapt to changing market conditions with RL agents.

## Installation

```bash
pip install stajax
```

## Quick Start

```python
import stajax as sj

# Create a market graph
market_graph = sj.MarketGraph(assets=['AAPL', 'GOOGL', 'MSFT'])

# Run causal discovery
causal_model = sj.CausalDiscovery(market_graph)

# Develop a trading strategy
strategy = sj.EvolutionaryStrategy(causal_model)

# Backtest the strategy
results = sj.backtest(strategy, start_date='2020-01-01', end_date='2021-12-31')

# Print results
print(results.summary())
```

## Documentation

For detailed documentation, please visit [our documentation site](https://stajax.readthedocs.io).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

`stajax` is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Citing

If you use `stajax` in your research, please cite:

```
@software{stajax2023,
  author = {{CausalQuant AI}},
  title = {stajax: Causal Quantitative Trading with JAX},
  year = {2023},
  url = {https://github.com/causalquant/stajax}
}
```

## Contact

For any questions or feedback, please open an issue on GitHub or contact us at support@causalquant.ai.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be used for live trading. Always consult with a qualified financial advisor before making any investment decisions.
