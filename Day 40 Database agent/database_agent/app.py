"""Quick manual smoke-test for the main agent (not the API).

The SQL sub-agent now borrows its own pooled connection internally, so no
connection needs to be passed in - only a thread_id for checkpointing.

    python app.py
"""

from dotenv import load_dotenv

from agent.main_agent.graph import agent

load_dotenv()


def main() -> None:
    config = {"configurable": {"thread_id": "smoke-test"}}
    user_input = "Find the top 5 record based on the bill amount"

    result = agent.invoke({"user_question": user_input}, config=config)

    print("Summarized:", result.get("summarized_output"))
    print("Answers:", result.get("final_answer"))


if __name__ == "__main__":
    main()
