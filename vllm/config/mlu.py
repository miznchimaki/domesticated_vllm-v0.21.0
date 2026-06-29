# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any, Literal
from pydantic import Field, ConfigDict, field_validator, ValidationInfo
from vllm.config.utils import config
from vllm.platforms import current_platform
from vllm.logger import init_logger

logger = init_logger(__name__)

QKV_DTYPE = Literal['auto', 'int8', 'fp8', 'fp8_e4m3']
SCALE_FMT = Literal['auto', 'e8m0', 'bfloat16']
ATTN_COMPUTE_DTYPE = Literal['auto', 'float32', 'float16', 'bfloat16']

@config(config=ConfigDict(arbitrary_types_allowed=True))
class MluAttentionConfig:
    """Configuration for attention."""
    q: QKV_DTYPE = 'auto'
    """ attention q dtype """
    k: QKV_DTYPE = 'auto'
    """ attention k dtype """
    v: QKV_DTYPE = 'auto'
    """ attention v dtype """
    scale_fmt_q: SCALE_FMT = 'auto'
    """ attention q scale format """
    scale_fmt_k: SCALE_FMT = 'auto'
    """ attention k scale format """
    scale_fmt_v: SCALE_FMT = 'auto'
    """ attention v scale format """
    compute_dtype: ATTN_COMPUTE_DTYPE = 'auto'
    """ attention compute dtype """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        factors.append(self.q)
        factors.append(self.k)
        factors.append(self.v)
        factors.append(self.scale_fmt_q)
        factors.append(self.scale_fmt_k)
        factors.append(self.scale_fmt_v)
        factors.append(self.compute_dtype)
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    @classmethod
    def from_dict(cls, dict_value: dict) -> "MluAttentionConfig":
        """Parse the CLI value for the Attention config."""
        return cls(**dict_value)

    @field_validator('scale_fmt_q', 'scale_fmt_k', 'scale_fmt_v', mode="after")
    @classmethod
    def validate_qkv_scales(
        cls, qkv_scales: SCALE_FMT, info: ValidationInfo
    ) -> SCALE_FMT:
        scale_field_name = info.field_name
        field_name = scale_field_name[-1]
        qkv = info.data.get(field_name, 'auto')

        if qkv_scales != 'auto' and qkv == 'auto':
            raise ValueError(
                f"scale format {qkv_scales} (from {scale_field_name}) is only supported when "
                f"qkv dtype is 'int8', 'fp8' or 'fp8_e4m3', got {field_name}={qkv}"
            )
        if qkv_scales != 'auto' or qkv.startswith("fp8"):
            if current_platform.is_out_of_tree() and current_platform.get_device_capability().major < 6:
                raise ValueError(f"MLU backend does not support q/k/v dtype: {qkv} for now")
        return qkv_scales

    @property
    def is_default(self) -> bool:
        qkv_args = [self.q, self.k, self.v]
        qkv_scales = [self.scale_fmt_q, self.scale_fmt_k, self.scale_fmt_v]
        all_args = qkv_args + qkv_scales + [self.compute_dtype]
        if all(arg == 'auto' for arg in all_args):
            return True
        return False

    @property
    def attn_dtype(self) -> str | None:
        qkv_args = [self.q, self.k, self.v]
        qkv_scales = [self.scale_fmt_q, self.scale_fmt_k, self.scale_fmt_v]
        all_args = qkv_args + qkv_scales + [self.compute_dtype]
        if all(arg == 'auto' for arg in all_args):
            return None
        if all(arg == qkv_args[0] for arg in qkv_args) and self.compute_dtype == 'auto':
            if all(arg == 'auto' for arg in qkv_scales):
                return qkv_args[0]
        return "customized"

    def init_qkv(self, dtype: str):
        self.q = dtype
        self.k = dtype
        self.v = dtype
        self.scale_fmt_q = 'auto'
        self.scale_fmt_k = 'auto'
        self.scale_fmt_v = 'auto'
        self.compute_dtype = 'auto'

@config(config=ConfigDict(arbitrary_types_allowed=True))
class MluConfig:
    """Configuration for vllm_mlu."""

    # common configs for all model

    prefill_enable_mlugraph: bool = False
    """Whether to use context mlugraph.

    When True, the system will capture a context mlugraph for the specific
    batch size and seq len.
    NOTE: Context mlugraph can only used in offline benchmark.
    """

    prefill_mlugraph_batch_size: int = None
    """The batch size that captured by context mlugraph."""

    prefill_mlugraph_seq_len: int = None
    """The seq len that captured by context mlugraph."""

    # specific configs for deepseek model

    dispatch_shared_expert_parallel: bool = True
    """
    Whether to parallelize the computation of shared-experts and the all-gather
    of the input to the MOE layer.
    """

    decode_dispatch_combine_use_all2all: bool = False
    """Whether to use All-to-All communication for combining dispatched results
    during decoding.

    When True, uses more aggressive All-to-All communication pattern which may
    improve performance in large-scale distributed decoding scenarios at the
    cost of higher network bandwidth usage. Defaults to False for most use
    cases.
    """

    layer_embedding_logit_tp_size: int | None = None
    """Tensor parallelism degree specifically for embedding layers.

    If None, will use the global tensor parallel size. Can be set to a smaller
    value than the global TP size to reduce memory usage for large embedding
    tables while maintaining higher parallelism for other layers.
    """

    layer_dense_mlp_tp_size: int | None = None
    """Tensor parallelism degree specifically for dense MLP layers.

    If None, will use the global tensor parallel size. Useful for models where
    MLP layers have different computational characteristics than attention
    layers, allowing for more optimal resource allocation.
    """

    prefill_dispatch_use_RS_AG: bool = True
    """Whether to use Reduce-Scatter and All-Gather for prefill phase dispatch.

    When True (default), uses optimized Reduce-Scatter/All-Gather communication
    pattern during the prefill phase which is generally more efficient for
    longer sequences. Set to False only for debugging or specific workload
    characteristics.
    """

    decoder_attn_dtype: QKV_DTYPE | None = None
    """Decoder attention dtype.

    If None, will use the global attention dtype.
    """

    prefill_use_sequence_parallel: bool = False
    """Whether to use sequence parallel when prefill.

    Note that this is ONLY supported when using deepseek v3.2 model.
    """
    enable_mamba_split_page_size: bool = False
    """Split the pages of Mamba so that the page size matches the page size of 
    full attention."""

    mamba_to_attn_block_ratio: int = 1
    """Defines the scaling factor between Mamba and full attention block sizes.
    Mamba typically operates with a block size several times larger than
    standard full attention."""

    mamba_support_max_batch_size: int = 1
    """Define a variable for the causal_conv1d_update function of qwen3-next to
    support batch parallel computation for this function."""

    indexer_online_quantization: bool = False
    """Define a variable for the indexer online quantization."""

    prefill_attn_config: MluAttentionConfig = Field(default_factory=MluAttentionConfig)
    """Configuration for prefill attention."""

    decoder_attn_config: MluAttentionConfig = Field(default_factory=MluAttentionConfig)
    """Configuration for decoder attention."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        factors.append(self.prefill_attn_config.compute_hash())
        factors.append(self.decoder_attn_config.compute_hash())
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    @classmethod
    def from_dict(cls, dict_value: dict) -> "MluConfig":
        """Parse the CLI value for the MLU config."""
        return cls(**dict_value)

    def __post_init__(self):
        """Verify configs are valid."""
        self._verify_args()
        if self.decoder_attn_dtype is not None and self.decoder_attn_config.attn_dtype is None:
            self.decoder_attn_config.init_qkv(self.decoder_attn_dtype)

    def _verify_args(self) -> None:
        if self.decoder_attn_dtype is not None and self.decoder_attn_config.attn_dtype is not None:
            if self.decoder_attn_dtype != self.decoder_attn_config.attn_dtype:
                raise ValueError(
                    f"Parameter conflict: decoder_attn_dtype({self.decoder_attn_dtype}) "
                    f"cannot be different from decoder_attn.attn_dtype({self.decoder_attn_config.attn_dtype})"
                )
