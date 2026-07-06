from agent.competitor_agent.graph import agent
from dotenv import load_dotenv


load_dotenv()

config = {"configurable" :{
    "thread_id" : "1"
}}
for i in agent.stream({"pitch_text" : "THe idea to create a work space for anit national prople spicially for indian people" ,
                       "pitch_summary" :"THe idea to create a work space for anit national prople spicially for indian people', 'pitch_summary': 'The proposal aims to establish a dedicated workspace specifically designed for anti-national individuals, with a particular focus on the Indian demographic. The project seeks to provide a specialized environment catering to this niche user group, though specific operational features and intended social or economic impacts remain to be defined."} , config=config):
    print(i) 



state = agent.get_state(config)

import re

def format_document(document: str):
    """Pretty print the tagged document."""

    # Remove document tags
    document = document.replace("<D>", "").replace("</D>", "")

    # Headings
    document = re.sub(
        r"<H1>(.*?)</H1>",
        lambda m: f"\n{'='*80}\n{m.group(1).strip().upper()}\n{'='*80}\n",
        document,
        flags=re.DOTALL,
    )

    document = re.sub(
        r"<H2>(.*?)</H2>",
        lambda m: f"\n\n{'-'*80}\n{m.group(1).strip()}\n{'-'*80}\n",
        document,
        flags=re.DOTALL,
    )

    document = re.sub(
        r"<H3>(.*?)</H3>",
        lambda m: f"\n▶ {m.group(1).strip()}\n",
        document,
        flags=re.DOTALL,
    )

    # Paragraphs
    document = re.sub(
        r"<P>(.*?)</P>",
        lambda m: m.group(1).strip() + "\n",
        document,
        flags=re.DOTALL,
    )

    # Lists
    def list_formatter(match):
        lines = match.group(1).strip().splitlines()
        formatted = []
        for line in lines:
            line = line.strip()
            if line.startswith("-"):
                formatted.append(f"  • {line[1:].strip()}")
            else:
                formatted.append(f"  • {line}")
        return "\n".join(formatted) + "\n"

    document = re.sub(
        r"<L>(.*?)</L>",
        list_formatter,
        document,
        flags=re.DOTALL,
    )

    # Tables
    def table_formatter(match):
        rows = [
            [c.strip() for c in row.split("|")]
            for row in match.group(1).strip().splitlines()
        ]

        if not rows:
            return ""

        widths = [
            max(len(r[i]) for r in rows)
            for i in range(len(rows[0]))
        ]

        output = []

        for i, row in enumerate(rows):
            output.append(
                " | ".join(
                    cell.ljust(widths[j]) for j, cell in enumerate(row)
                )
            )
            if i == 0:
                output.append("-+-".join("-" * w for w in widths))

        return "\n".join(output) + "\n"

    document = re.sub(
        r"<T>(.*?)</T>",
        table_formatter,
        document,
        flags=re.DOTALL,
    )

    # Sources
    document = re.sub(
        r"<S>",
        "\n\nSOURCES\n" + "=" * 80 + "\n",
        document,
    )
    document = document.replace("</S>", "")

    # Citations
    document = re.sub(r"<C>(.*?)</C>", r" \1", document)

    # Cleanup
    document = re.sub(r"\n{3,}", "\n\n", document)

    print(document.strip())

format_document(state.values["report"])