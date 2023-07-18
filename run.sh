ulimit -n 64000;
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py model=dolly7b \
    datasets=[dolly] loss=dpo loss.beta=0.1 \
    exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=8 \
    batch_size=64 eval_batch_size=32 trainer=FSDPTrainer \
    sample_during_eval=false model.fsdp_policy_mp=bfloat16 \
    model.name_or_path=databricks/dolly-v2-7b \
    wandb.enabled=False