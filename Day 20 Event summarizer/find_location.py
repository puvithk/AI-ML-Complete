class IntermediateInfo(Event):
    result: str | dict

class LoopEvent(Event):
    result : str | dict 

class MultiAgentWorkflow(Workflow):



    @step
    async def get_calendar_details(self, start: StartEvent | LoopEvent) -> IntermediateInfo | LoopEvent:
        now = datetime.now()
        today_str = now.strftime("%B %d, %Y")

        last_day = calendar.monthrange(now.year, now.month)[1]
        end_of_month_str = now.replace(day=last_day).strftime("%B %d, %Y")

        query = f"""
        I need a summary of my schedule.

        Today's date is {today_str}.

        Tasks:
        1. List today's meetings ({today_str})
        2. List important non-meeting events until {end_of_month_str}
           (deadlines, exams, submissions, milestones)
        """
        try:
            response = await calendar_agent.run(query)
        except Exception as e :
            return LoopEvent(result="Error Correcting " +  e)
        print("Agent Response:\n", response)

        return IntermediateInfo(result=response)
    async def create_send_mail(self , ev : IntermediateInfo) -> StopEvent:

        pass


multi_agent = MultiAgentWorkflow()