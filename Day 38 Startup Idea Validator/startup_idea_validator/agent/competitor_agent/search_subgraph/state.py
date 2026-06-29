from pydantic import BaseModel , Field 
from typing import Annotated , TypedDict  
import operator


class WebSearchResultState(TypedDict):


    question : str

    raw_data : Annotated[list[str] , operator.add]

    cleaned_data  : Annotated[list[str] , operator.add]
    sources: Annotated[list[dict], operator.add]


class Source(BaseModel):
    title : str = Field(description='Title of the source')
    url: str = Field(description="Source URL")
    content: str = Field(description="Cleaned content extracted from the source")


class WebSearchedCleanedResult(BaseModel):
    source : list[Source] = Field(description="Contains the sources and complete cleaned data ")