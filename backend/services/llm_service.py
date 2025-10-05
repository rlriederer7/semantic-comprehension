import os
from typing import List

from anthropic import AsyncAnthropic


class AnthropicProvider():
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"

    async def generate(self, prompt: str, context_chunks: List[tuple[str, str]]) -> str:
        print(context_chunks)
        context = "\n\n---\n\n".join(f'Document Name: {src}\nText: "{text}"' for text, src in context_chunks)

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=(
                    "The user is querying for something in a list of documents. "
                    "Tell the user where they can find what they are looking for in the context documents, if it exists. "
                    "Provide both the document name and a short quote that answers their query. "
                    "For instance, if the user queries 'senior python dev' and the context indicates that document 'JohnSmithResume01' has '8yrs python experience' "
                    "you should tell the user that 'The document JohnSmithResume01 lists '8yrs python experience'"
                    "Only use information from the context. "
                    "If the answer is not in the context, say so."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": f"Context Documents:\n{context}\n\nQuery: {prompt}\n\nAnswer: "
                    }
                ],
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic error: {e}")
            raise

llm_service = AnthropicProvider()