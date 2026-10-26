from akas.models.spark import Spark
from akas.compilation import compile_checkpoint
from akas.compilation.tuning import SearchSettings
compile_checkpoint(Spark,
 '/home/sagemaker-user/akas/build/checkpoints/spark-x2.6-4b/gguf/Spark-X2.6-4B-Q4_K_M.gguf',
 '/home/sagemaker-user/akas/build/spark-parity/compact-program', checkpoint_options={'tokenizer':False},
 geometries=[(1,1),(1,32),(1,512),(1,2048)],max_requests=1,max_context=131072,
 model_options=dict(prefill_chunk=2048,rotary_cache=True,optimized_attention=True,embedding_location='host',weight_arena=True,fused_ffn=True,half_activations=True,compact_swa=True),
 search_settings=SearchSettings(budget=1,samples=1,repeats=1))
