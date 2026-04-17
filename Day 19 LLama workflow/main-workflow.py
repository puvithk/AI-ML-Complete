from llama_index.core.workflow import StartEvent , StopEvent , Event , Workflow , step

class IntermidiateResult(Event):
    intermidiate_result : str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self , ev: StartEvent) -> IntermidiateResult:
        return IntermidiateResult(intermidiate_result="Step one Started")

    @step
    async def last_step(self , ev:IntermidiateResult) -> StopEvent:
        final_result = f'Finised procession : {ev.intermidiate_result}'
        return StopEvent(result=final_result)
    
async def main():
    workflow = MultiStepWorkflow(timeout=10 , verbose=False)
    result = await workflow.run()
    print(result)
import asyncio
if __name__ == "__main__":
    asyncio.run(main())