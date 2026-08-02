(references/api/exceptions)=

# Exceptions

The framework exception hierarchy. Every exception derives from
`TroopAIError`, so a single `except TroopAIError` catches all
framework-raised failures.

## Base

```{eval-rst}
.. autoclass:: troopai.adk.exceptions.TroopAIError
   :members:
   :show-inheritance:
```

## Concrete exceptions

```{eval-rst}
.. autoclass:: troopai.adk.exceptions.AgentInputGuardrailTripwireTriggered
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.AgentOutputGuardrailTripwireTriggered
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.AgentToolDeferral
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ApplyPatchError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.CheckpointConflictError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ConfigError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ConfigParseError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ConfigResolutionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.DocumentLoadError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ExecFailureError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ExecNonZeroError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ExecTimeoutError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ExecTransportError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ExposedPortUnavailableError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.GitArtifactError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.GraphNodeTimeoutError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.GuardrailTripwireTriggered
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.HandoffDefinitionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.HandoffRejection
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.InvalidCompressionSchemeError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.InvalidManifestPathError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.LocalArtifactError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.MaxTurnsExceeded
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.MemoryExtractionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ModelRefusalError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.MountArtifactError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.NoRoutingCandidateError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.NodeRetriesExhaustedError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.PtySessionNotFoundError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.QuotaExceeded
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxArtifactError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxCommandRejected
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxConcurrencyError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxConfigurationError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxNetworkPolicyViolation
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxResourceLimitExceeded
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxRuntimeError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxSelectionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxStartFailed
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SandboxStopFailed
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SessionAppendConflictError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SkillsConfigError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SnapshotError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SnapshotNotRestorableError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SnapshotPersistError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.SnapshotRestoreError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.TenantBudgetExceeded
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ToolDependencyError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ToolGuardrailTripwireTriggered
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ToolNotPermittedForTenant
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ToolRetry
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ToolTimeoutError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.ToolsetNameConflictError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.TracingDependencyError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UnsupportedDocumentSourceError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UnsupportedManifestEntryError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UnsupportedMountPatternError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UnsupportedMountStrategyError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UnsupportedSandboxClientError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UnsupportedSnapshotFeatureError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UsageLimitExceeded
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.UserError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.WorkspaceArchiveReadError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.WorkspaceArchiveWriteError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.WorkspaceIOError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.exceptions.WorkspaceReadNotFoundError
   :members:
   :show-inheritance:
```

Domain-specific exceptions also live next to their modules — see
{ref}`references/api/flows` (flow execution), {ref}`references/api/mcp`,
and {ref}`references/api/a2a`.
