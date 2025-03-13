import streamlit as st

st.set_page_config(layout="wide")

import nest_asyncio

nest_asyncio.apply()  # Required for streamlit to work with asyncio.

from utils import add_sink_to_logger

add_sink_to_logger()

from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)
from your_agent import YourAgent
from dotenv import load_dotenv

load_dotenv()


def app() -> None:
    st.title("Agent's decision-making process")

    # Fetch markets from the selected market type.
    market_source = MarketType(
        st.selectbox(
            "Select a market source",
            [market_source.value for market_source in MarketType],
        )
    )
    markets = get_binary_markets(42, market_source)

    # Ask the user to provide a question.
    custom_question_input = st.checkbox("Provide a custom question", value=False)
    question = (
        st.text_input("Question")
        if custom_question_input
        else st.selectbox("Select a question", [m.question for m in markets])
    )
    if not question:
        st.warning("Please enter a question.")
        st.stop()

    market = (
        [m for m in markets if m.question == question][0]
        if not custom_question_input
        # If custom question is provided, just take some random market and update its question.
        else markets[0].model_copy(update={"question": question, "current_p_yes": 0.5})
    )

    if not custom_question_input:
        st.info(
            f"Current probability {market.current_p_yes * 100:.2f}% at {market.url}."
        )

    agent = YourAgent()
    result = agent.answer_binary_market(market)

    st.markdown(f"{agent.__class__.__name__}'s answer is {result}.")


if __name__ == "__main__":
    app()
