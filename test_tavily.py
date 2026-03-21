from tavily import TavilyClient

client = TavilyClient(api_key="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")
response = client.search("test query")
print(response)
