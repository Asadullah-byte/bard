from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

class JournalGenerator:
    def __init__(self):
        self.vision_llm = ChatOpenAI(
            model=settings.OPENAI_VISION_MODEL_NAME, 
            api_key=settings.OPENAI_API_KEY,
            max_tokens=300
        )
        self.text_llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_NAME, 
            api_key=settings.OPENAI_API_KEY,
            temperature=0.7
        )
        
        self.entry_prompt = ChatPromptTemplate.from_messages([
           ("system",settings.SYSTEM_DAY_JOURNAL_PROMPT),
            ("human", f"{settings.USER_DAY_JOURNAL_PROMPT}\n\nDescriptions:\n{{descriptions}}")
        ])
        self.entry_chain = self.entry_prompt | self.text_llm | StrOutputParser()

    async def describe_image(self, base64_image: str, metadata_str: str = "") -> str:
        """
        Generates a description for a single image using OpenAI Vision.
        """
        msg = [
            SystemMessage(content=settings.SYSTEM_DESCRIBE_IMAGE_PROMPT)
            ,HumanMessage(
            content=[
                {"type": "text", "text": f"{settings.USER_DESCRIBE_IMAGE_PROMPT} {metadata_str}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )]
        response = await self.vision_llm.ainvoke(msg)
        return response.content

    async def generate_entry(self, descriptions: List[str]) -> str:
        """
        Generates a journal entry from a list of image descriptions.
        """
        descriptions_text = "\n".join([f"- {desc}" for desc in descriptions])
        return await self.entry_chain.ainvoke({"descriptions": descriptions_text})
