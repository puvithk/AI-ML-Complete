from dotenv import load_dotenv
import os

load_dotenv(override=True)
print(os.environ.get("LANGSMITH_TRACING"))  # should print "true"