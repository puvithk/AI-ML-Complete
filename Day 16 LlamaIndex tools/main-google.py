from llama_index.tools.google import GmailToolSpec
tools = GmailToolSpec()
tool_spec_list = tools.to_tool_list()

print([(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list])