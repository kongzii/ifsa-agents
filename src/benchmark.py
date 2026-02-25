# from advanced_agent import AdvancedAgent
# from basic_agent import BasicAgent
# from your_agent import YourAgent
# from openai_search_agent_high import OpenaiSearchAgentHigh
from openai_search_agent_variable import OpenaiSearchAgentVariable

from prediction_market_agent_tooling.benchmark.agents import (
    AbstractBenchmarkedAgent,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction,
    Prediction,
)
from datetime import timedelta

from web3 import Web3

from prediction_market_agent_tooling.gtypes import HexBytes
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.data_models import Condition
from prediction_market_agent_tooling.markets.omen.omen import (
    MarketFees,
    OmenAgentMarket,
)
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    WrappedxDaiContract,
)
from prediction_market_agent_tooling.tools.utils import utcnow

import typing as t

import typer
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
    FilterBy,
    SortBy,
)


def main(
    n: int = 10,
    output: str = "./benchmark_report.md",
    cache_path: t.Optional[str] = None,
    only_cached: bool = False,
) -> None:
    markets = get_binary_markets(
        n, MarketType.MANIFOLD, filter_by=FilterBy.OPEN, sort_by=SortBy.NONE
    )
    markets_deduplicated = list(({m.question: m for m in markets}.values()))

    print(f"Found {len(markets_deduplicated)} markets.")

    agents = [
        # BenchmarkAgent(agent=AdvancedAgent()),
        # BenchmarkAgent(agent=YourAgent()),
        # BenchmarkAgent(agent=OpenaiSearchAgentHigh())
    ]

    settings = ["high"]

    # Create additional agents with nested loops.
    for reasoning in settings:
        for search_context_size in settings:
            # Instantiate the agent without parameters.
            agent_instance = OpenaiSearchAgentVariable()
            # Call the load method to set up the custom parameters.
            agent_instance.load(reasoning=reasoning, search_context_size=search_context_size)
            agents.append(BenchmarkAgent(agent=agent_instance))

    # Now pass the full list to your Benchmarker.
    benchmarker = Benchmarker(
        markets=markets_deduplicated,
        agents=agents,
        cache_path=cache_path,
        only_cached=only_cached,
    )


    # benchmarker = Benchmarker(
    #     markets=markets_deduplicated,
    #     agents=[
    #         BenchmarkAgent(agent=AdvancedAgent()),
    #         BenchmarkAgent(agent=BasicAgent()),
    #         # BenchmarkAgent(agent=YourAgent()), 
    #         BenchmarkAgent(agent=OpenaiSearchAgentHigh()),
            
    #         for web_search in settings:
    #             for reasoning in settings:
    #                 BenchmarkAgent(agent=OpenaiSearchAgentVariable(web_search, reasoning)), 
    #     ],
    #     cache_path=cache_path,
    #     only_cached=only_cached,
    # )

    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)


class BenchmarkAgent(AbstractBenchmarkedAgent):
    def __init__(self, agent: DeployableTraderAgent) -> None:
        # Use the custom name if set, otherwise fallback on the class name.
        agent_name = getattr(agent, "custom_agent_name", agent.__class__.__name__)
        super().__init__(agent_name=agent_name, max_workers=1)
        self.agent = agent

    def predict(self, market_question: str) -> Prediction:
        try:
            output = self.agent.answer_binary_market(
                market=OmenAgentMarket(
                    question=market_question,
                    # Rest of the fields are not used in this benchmark, so we can just fill them with dummy values.
                    id="id",
                    creator="creator",  # type: ignore # dummy input
                    outcomes=["Yes", "No"],
                    current_p_yes=0.5,  # type: ignore # dummy input
                    collateral_token_contract_address_checksummed=WrappedxDaiContract().address,
                    market_maker_contract_address_checksummed=Web3.to_checksum_address(
                        "0xf3318C420e5e30C12786C4001D600e9EE1A7eBb1"
                    ),
                    created_time=utcnow() - timedelta(days=1),
                    close_time=utcnow(),
                    resolution=None,
                    condition=Condition(id=HexBytes("0x123"), outcomeSlotCount=2),
                    url="url",
                    volume=None,
                    finalized_time=None,
                    fees=MarketFees.get_zero_fees(bet_proportion=0.02),
                    outcome_token_pool=None,
                )
            )
        except ValueError as e:
            print(f"Failed to predict for market by {self.agent_name}: {e}")
            output = None
        if output is None:
            print(
                f"Failed to predict for market by {self.agent_name}: {market_question}"
            )
            return Prediction()
        return Prediction(
            is_predictable=True,
            outcome_prediction=OutcomePrediction(
                p_yes=output.p_yes,
                confidence=output.confidence,
                info_utility=None,
            ),
        )


if __name__ == "__main__":
    typer.run(main)
