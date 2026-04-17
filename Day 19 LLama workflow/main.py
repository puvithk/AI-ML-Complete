from llama_index.core.workflow import StartEvent , StopEvent , Workflow , step 

class MyWorkFlow(Workflow):
    @step
    async  def my_step(self , ev : StartEvent) -> StopEvent:
        return StopEvent(result="Hello  welcome")

workflow = MyWorkFlow(timeout=10 , verbose=False)
async def main():
    result = await workflow.run()
    print(result)
import asyncio
if __name__ == "__main__":
    asyncio.run(main())