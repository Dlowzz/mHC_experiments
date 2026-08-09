"""mHC hyper-connection variants kept for the main experiments.

Selectable ``hyper_conn_type``:

  ``none``                       plain residual, no hyper-connections
  ``mhc``                        baseline mHC
  ``mhc_lite``                   mHC-Lite
  ``mhc_group_embedding``        mHC-group (group-wise H_pre read)
  ``mhc_lora_residual_midnorm``  mHC-LoRA   (midnorm + in-beta)
  ``mhc_group_lora_midnorm``     mHC-group-LoRA (midnorm + in-beta)

Ablations of the two group variants (dense n^3*C read for H_pre instead of the
group-local n^2*C read):

  ``mhc_group_dense_embedding``      mHC-group with dense H_pre
  ``mhc_group_lora_dense_midnorm``   mHC-group-LoRA with dense H_pre

Ablations of the LoRA norm (all on ``mhc_lora_residual_midnorm``):

  ``mhc_lora_residual_scalemidnorm`` rank-dim norm + learnable scale (no bias)
  ``mhc_lora_residual_prenorm``      norm moved before A_s (on h, hidden dim)
  ``mhc_lora_residual_postnorm``     norm moved after B_s (on delta, hidden dim)

The same three norm ablations on the group-LoRA line (group-wise H_pre read kept):

  ``mhc_group_lora_scalemidnorm``    rank-dim norm + learnable scale (no bias)
  ``mhc_group_lora_prenorm``         norm moved before A_s (on h, hidden dim)
  ``mhc_group_lora_postnorm``        norm moved after B_s (on delta, hidden dim)

``mhc_lora_residual.py`` and ``mhc_group_lora.py`` are kept because the two midnorm
variants inherit from them; the no-norm variants themselves are no longer selectable.
Every other variant (hc / mhc_embedding / mhc_orthogonal_diff / mhc_group_lora_capped /
mhc_lora_residual_affinemidnorm / analysis) has been removed from this branch -- use the
`final` branch to train or evaluate their old checkpoints.
"""
from .mhc import (
    ManifoldConstrainedHyperConnections,
    Residual,
    StreamEmbed,
    AttentionPoolReduceStream,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mc_get_init_and_expand_reduce_stream_functions,
)

from .mhc_lite import (
    MHCLite,
    get_init_and_expand_reduce_stream_functions as mhclite_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_embedding import (
    ManifoldConstrainedHyperConnectionsGroupEmbedding,
    MHCGroupEmbedding,
    get_init_and_expand_reduce_stream_functions as mhc_group_embedding_get_init_and_expand_reduce_stream_functions,
)

from .mhc_lora_residual_midnorm import (
    ManifoldConstrainedHyperConnectionsLoRAResidualMidNorm,
    MHCLoRAResidualMidNorm,
    get_init_and_expand_reduce_stream_functions as mhc_lora_residual_midnorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_lora_midnorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRAMidNorm,
    MHCGroupLoRAMidNorm,
    get_init_and_expand_reduce_stream_functions as mhc_group_lora_midnorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_dense_embedding import (
    ManifoldConstrainedHyperConnectionsGroupDenseEmbedding,
    MHCGroupDenseEmbedding,
    get_init_and_expand_reduce_stream_functions as mhc_group_dense_embedding_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_lora_dense_midnorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRADenseMidNorm,
    MHCGroupLoRADenseMidNorm,
    get_init_and_expand_reduce_stream_functions as mhc_group_lora_dense_midnorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_lora_residual_scalemidnorm import (
    ManifoldConstrainedHyperConnectionsLoRAResidualScaleMidNorm,
    MHCLoRAResidualScaleMidNorm,
    get_init_and_expand_reduce_stream_functions as mhc_lora_residual_scalemidnorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_lora_residual_prenorm import (
    ManifoldConstrainedHyperConnectionsLoRAResidualPreNorm,
    MHCLoRAResidualPreNorm,
    get_init_and_expand_reduce_stream_functions as mhc_lora_residual_prenorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_lora_residual_postnorm import (
    ManifoldConstrainedHyperConnectionsLoRAResidualPostNorm,
    MHCLoRAResidualPostNorm,
    get_init_and_expand_reduce_stream_functions as mhc_lora_residual_postnorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_lora_scalemidnorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRAScaleMidNorm,
    MHCGroupLoRAScaleMidNorm,
    get_init_and_expand_reduce_stream_functions as mhc_group_lora_scalemidnorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_lora_prenorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRAPreNorm,
    MHCGroupLoRAPreNorm,
    get_init_and_expand_reduce_stream_functions as mhc_group_lora_prenorm_get_init_and_expand_reduce_stream_functions,
)

from .mhc_group_lora_postnorm import (
    ManifoldConstrainedHyperConnectionsGroupLoRAPostNorm,
    MHCGroupLoRAPostNorm,
    get_init_and_expand_reduce_stream_functions as mhc_group_lora_postnorm_get_init_and_expand_reduce_stream_functions,
)

SUPPORTED_HYPER_CONN_TYPES = (
    "none",
    "mhc",
    "mhc_lite",
    "mhc_group_embedding",
    "mhc_lora_residual_midnorm",
    "mhc_group_lora_midnorm",
    # ablations: dense (n^3*C) H_pre read for the group variants
    "mhc_group_dense_embedding",
    "mhc_group_lora_dense_midnorm",
    # ablations: LoRA norm affine / position
    "mhc_lora_residual_scalemidnorm",
    "mhc_lora_residual_prenorm",
    "mhc_lora_residual_postnorm",
    # the same three norm ablations on the group-LoRA line
    "mhc_group_lora_scalemidnorm",
    "mhc_group_lora_prenorm",
    "mhc_group_lora_postnorm",
)

flag = False

def hyper_conn_init_func(hyper_conn_type: str, hyper_conn_n: int):
    global flag
    if not flag:
        print(f"HYPER_CONN: USING {hyper_conn_type} with {hyper_conn_n} streams")
        flag = True

    if hyper_conn_type == "none":
        # plain residual: the same (Residual, Identity, Identity) triple the removed
        # hyper_connections.py returned for disable=True
        return mc_get_init_and_expand_reduce_stream_functions(hyper_conn_n, disable = True)
    elif hyper_conn_type == "mhc":
        return mc_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_lite":
        return mhclite_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_embedding":
        return mhc_group_embedding_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_lora_residual_midnorm":
        return mhc_lora_residual_midnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_lora_midnorm":
        return mhc_group_lora_midnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_dense_embedding":
        return mhc_group_dense_embedding_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_lora_dense_midnorm":
        return mhc_group_lora_dense_midnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_lora_residual_scalemidnorm":
        return mhc_lora_residual_scalemidnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_lora_residual_prenorm":
        return mhc_lora_residual_prenorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_lora_residual_postnorm":
        return mhc_lora_residual_postnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_lora_scalemidnorm":
        return mhc_group_lora_scalemidnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_lora_prenorm":
        return mhc_group_lora_prenorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc_group_lora_postnorm":
        return mhc_group_lora_postnorm_get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    else:
        raise ValueError(
            f"Invalid hyper connection type: {hyper_conn_type}. supported: "
            f"{', '.join(SUPPORTED_HYPER_CONN_TYPES)} "
            f"(the other variants live on the `final` branch)"
        )
