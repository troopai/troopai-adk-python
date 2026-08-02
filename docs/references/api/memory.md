(references/api/memory)=

# Memory

Extracted, searchable knowledge carried across sessions.

## Core

```{eval-rst}
.. autoclass:: troopai.adk.memory.Memory
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemoryConfig
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemoryInjectionPosition
   :members:
   :show-inheritance:
```

## Entries and search

```{eval-rst}
.. autoclass:: troopai.adk.memory.MemoryEntry
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemoryKind
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemoryMetadata
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemorySource
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemorySearchFilter
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.MemorySearchResult
   :members:
   :show-inheritance:
```

## Implementations

```{eval-rst}
.. autoclass:: troopai.adk.memory.TemporaryMemory
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.SQLiteMemory
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.VectorMemory
   :members:
   :show-inheritance:
```

## Vector stores

```{eval-rst}
.. autoclass:: troopai.adk.memory.VectorStore
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.VectorRecord
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.VectorQueryResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.InMemoryVectorStore
   :members:
   :show-inheritance:
```

## Extraction

```{eval-rst}
.. autoclass:: troopai.adk.memory.MemoryExtractor
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.LLMExtractor
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.memory.ExtractionResult
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.memory.distill_to_semantic
```

Usage lives in the [Memory guide](../../memory/memory.md).
