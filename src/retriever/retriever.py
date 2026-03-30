from src.storage.storage_manager import StorageManager

class Retriever:
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager

    async def get_context_for_llm(self, query: str, user_id: str) -> str:
        import asyncio
        docs_task = self.storage.retrieve(query=query, user_id=user_id, storage_type="document")
        history_task = self.storage.retrieve(query=query, user_id=user_id,storage_type="context")
    
        docs, history = await asyncio.gather(docs_task, history_task)

        context_parts = []

        if docs:
            doc_section = "### RELEVANT KNOWLEDGE (From Files)\n"
            for d in docs:
                source = d.get("metadata", {}).get("source", "Unknown")
                doc_section += f"[{source}]: {d['text']}\n"
            context_parts.append(doc_section)

        if history:
            hist_section = "### RELEVANT CHAT MEMORY (From Past Messages)\n"
            for h in history:
                hist_section += f"- {h['text']}\n"
            context_parts.append(hist_section)

        if not context_parts:
            return "No specific background information found for this query."

        return "\n\n".join(context_parts)