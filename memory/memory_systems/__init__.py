def _optional_import(module_name, class_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


MirixMemorySystem = _optional_import("mirix", "MirixMemorySystem")
LongContextMemorySystem = _optional_import("long_context", "LongContextMemorySystem")
Mem0MemorySystem = _optional_import("mem0", "Mem0MemorySystem")
LettaMemorySystem = _optional_import("letta", "LettaMemorySystem")
RAGMemorySystem = _optional_import("rag", "RAGMemorySystem")
MemoRAGMemorySystem = _optional_import("memorag", "MemoRAGMemorySystem")
GraphRAGMemorySystem = _optional_import("langchain_graphrag", "GraphRAGMemorySystem")
AMemMemorySystem = _optional_import("amem", "AMemMemorySystem")
LightMemMemorySystem = _optional_import("lightmem", "LightMemMemorySystem")
ReasoningBankMemorySystem = _optional_import("reasoningbank", "ReasoningBankMemorySystem")
ZepMemorySystem = _optional_import("zep", "ZepMemorySystem")

__all__ = [
    "MirixMemorySystem",
    "LongContextMemorySystem",
    "Mem0MemorySystem",
    "LettaMemorySystem",
    "RAGMemorySystem",
    "MemoRAGMemorySystem",
    "GraphRAGMemorySystem",
    "AMemMemorySystem",
    "LightMemMemorySystem",
    "ReasoningBankMemorySystem",
    "ZepMemorySystem",
]
