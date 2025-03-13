from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.loggers import logger


class YourAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        logger.info("This should be shown in the streamlit app.")
        logger.success("This as well, but in green!")
        raise NotImplementedError("Complete this.")


if __name__ == "__main__":
    agent = YourAgent()
    agent.run(market_type=MarketType.OMEN)
