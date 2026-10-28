"""Bounded draft upload with borrowed target embeddings and vocabulary weights."""

from math import prod
from akas.checkpoints.gguf import GGUF, GGUFError
from akas.checkpoints.upload import upload_packed
from akas.models.dspark.config import DSparkDraftConfig
from akas.quantization.weights.ggml.definition import weight_storage


def tensor_shapes(config):
    c, f, d = config.width, config.ff, config.head_dim
    kv = config.kv_heads * d
    result = {
        "fc.weight": (c, config.feature_width),
        "enc.output_norm.weight": (c,),
        "output_norm.weight": (c,),
        "markov_w1.weight": (config.vocab, config.markov_rank),
        "markov_w2.weight": (config.vocab, config.markov_rank),
        "conf_proj.weight": (1, c + config.markov_rank),
        "conf_proj.bias": (1,),
    }
    for layer in range(config.depth):
        prefix = f"blk.{layer}."
        result.update(
            {
                prefix + n: shape
                for n, shape in {
                    "attn_norm.weight": (c,),
                    "attn_q.weight": (c, c),
                    "attn_k.weight": (kv, c),
                    "attn_v.weight": (kv, c),
                    "attn_q_norm.weight": (d,),
                    "attn_k_norm.weight": (d,),
                    "attn_output.weight": (c, c),
                    "ffn_norm.weight": (c,),
                    "ffn_gate.weight": (f, c),
                    "ffn_up.weight": (f, c),
                    "ffn_down.weight": (c, f),
                }.items()
            }
        )
    return result


class DSparkDraftCheckpoint:
    def __init__(self, path, *, target):
        self.target = target
        self.target_checkpoint = getattr(target, "checkpoint", target)
        if not hasattr(self.target_checkpoint, "gguf"):
            raise ValueError("DSpark currently pairs with an original GGUF target")
        self.gguf = GGUF(path)
        self.config = DSparkDraftConfig.from_gguf(
            self.gguf, self.target_checkpoint.config
        )
        expected = tensor_shapes(self.config)
        if expected.keys() != self.gguf.tensors.keys():
            raise GGUFError("DSpark sidecar has missing or unsupported tensors")
        for name, shape in expected.items():
            info = self.gguf.tensors[name]
            if info.shape != shape:
                raise GGUFError(f"{name}: DSpark tensor dimensions differ")
            if len(shape) == 1 and info.type != 0:
                raise GGUFError(
                    f"{name}: DSpark requires native floating norm and bias weights"
                )
            if len(shape) > 1:
                if info.type not in (0, 1, 8):
                    raise GGUFError(f"{name}: DSpark matrices require F32, F16 or Q8_0")
                try:
                    weight_storage(info.type, shape[-1])
                except ValueError as error:
                    raise GGUFError(
                        f"{name}: invalid draft weight row encoding"
                    ) from error
        names = self.target_checkpoint.gguf.tensors
        head = "output.weight" if "output.weight" in names else "token_embd.weight"
        self.shared = {"embedding": "token_embd.weight", "head": head}
        for name in set(self.shared.values()):
            if names[name].shape != (self.config.vocab, self.config.width):
                raise GGUFError("DSpark target embedding/head dimensions differ")
        self.tokenizer = getattr(self.target_checkpoint, "tokenizer", None)

    def graph_weights(self, graph, *, provider):
        return DraftWeights(graph, self, provider=provider)

    def program_upload_options(self, weights, *, model_options):
        return {}

    def upload(self, device):
        shared = getattr(self.target, "weights", None)
        if shared is None:
            raise ValueError("DSpark execution requires its loaded target executor")
        for name in set(self.shared.values()):
            if name not in shared or shared[name].device is not device:
                raise ValueError(
                    "DSpark and target weights require the same device session"
                )
        weights = {"draft." + n: b for n, b in upload_packed(device, self.gguf).items()}
        try:
            for name in sorted(set(self.shared.values())):
                source = shared[name]
                # Releasing this descriptor never frees the target allocation.
                weights["target." + name] = source.view(source.shape)
            return weights
        except BaseException:
            for buffer in reversed(tuple(weights.values())):
                buffer.release()
            raise


class DraftWeights:
    def __init__(self, graph, checkpoint, *, provider):
        self.checkpoint = checkpoint
        self.infos = {
            **{"draft." + n: info for n, info in checkpoint.gguf.tensors.items()},
            **{
                "target." + n: checkpoint.target_checkpoint.gguf.tensors[n]
                for n in set(checkpoint.shared.values())
            },
        }
        self.buffers = {}
        for name, info in self.infos.items():
            count, dtype = weight_storage(
                info.type, prod(info.shape), packed_words=provider == "webgpu"
            )
            self.buffers[name] = graph.buffer(name, (count,), dtype, role="weight")

    def name(self, name):
        return (
            "target." + self.checkpoint.shared[name]
            if name in self.checkpoint.shared
            else "draft." + name
        )

    def __getitem__(self, name):
        return self.buffers[self.name(name)]

    def parameters(self, name):
        return dict(ggml=True, type=self.infos[self.name(name)].type)

    def matrix(self, name):
        key = self.name(name)
        return [(self.buffers[key], 0, self.infos[key].shape[0])]
