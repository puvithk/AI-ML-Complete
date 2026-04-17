from llama_index.core.workflow import Event , StartEvent , StopEvent, step , Workflow
from llama_index.utils.workflow import draw_all_possible_flows
import random

class IntermidiateEvent(Event):
    intermidiate_result : str


class LoopEvent(Event):
    loop_output :str


class MultiStepWorkFlow(Workflow):
    @step
    async def step_one(self , ev: StartEvent | LoopEvent) -> IntermidiateEvent | LoopEvent:
        if random.randint(0 , 1 ) == 0 :
            print("bad")
            return LoopEvent(loop_output="Back to step one function")
        else:
            print("Good ")
            return IntermidiateEvent(intermidiate_result="Complted step one")
    @step
    async def last_step(self , ev:IntermidiateEvent) -> StopEvent:
        return StopEvent(result=f'Finished {ev.intermidiate_result}')


async def main():
    workflow = MultiStepWorkFlow(timeout=10 , verbose=False)
    result = await workflow.run()
    draw_all_possible_flows(workflow)
    print(result)
    return result
import asyncio
if __name__ =="__main__":
    asyncio.run(main())
    