import importlib.util, json, time, pathlib
import torch
from transformers import GPTNeoXConfig
from huggingface_hub import snapshot_download
root=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_gpt_neox',root/'vendor/PonderLM-2/src/llamafactory/model/modeling/modeling_gpt_neox.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(42)
t=time.perf_counter()
p=snapshot_download('EleutherAI/pythia-70m',revision='a39f36b100fe8a5377810d56c3f4789b9c53ac42',allow_patterns=['config.json','model.safetensors'])
files={x.name:x.stat().st_size for x in pathlib.Path(p).iterdir() if x.is_file()}
print(json.dumps(dict(download_seconds=time.perf_counter()-t,files=files)),flush=True)
c=GPTNeoXConfig.from_pretrained(p)
for k,v in dict(num_latent_thoughts=1,num_jacobi_iterations=2,random_jacobi_iterations=False,scale_embeds=True,interpolation=False,recurrent_model=True,use_cache=False).items(): setattr(c,k,v)
t=time.perf_counter()
model=m.GPTNeoXForCausalLM.from_pretrained(p,config=c,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
print(json.dumps(dict(load_seconds=time.perf_counter()-t,parameters=sum(x.numel() for x in model.parameters()),gpu=torch.cuda.get_device_name(),total_bytes=torch.cuda.get_device_properties(0).total_memory)),flush=True)
# A bounded forward-only probe: train mode retains autograd graph, no backward or optimizer updates.
model.train()
for b,s in [(1,64),(1,256),(1,512),(2,256),(2,512)]:
 torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
 x=torch.randint(0,c.vocab_size,(b,s),device='cuda')
 torch.cuda.synchronize(); t=time.perf_counter()
 out=model(input_ids=x,labels=x,use_cache=False)
 torch.cuda.synchronize()
 print(json.dumps(dict(batch=b,seq=s,seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),loss=out.loss.item(),logits=list(out.logits.shape),finite=bool(torch.isfinite(out.logits).all()))),flush=True)
 del out,x
