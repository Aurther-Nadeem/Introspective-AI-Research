import fire
import logging
from dataclasses import dataclass
from introspectai.experiments.prefill_authorship import run_trial
from introspectai.utils.logging import setup_logging, JSONLLogger

@dataclass
class Config:
    model_name: str
    concept_store_path: str
    out_path: str
    layer: int = 10
    concept: str = "test"
    strength: float = 1.0
    condition: str = "self_generated" # self_generated, prefill_noinj, prefill_inj
    seed: int = 42
    max_new_tokens: int = 100

def main(
    model: str = "gpt2",
    concept_store: str = "datasets/concepts",
    out: str = "datasets/trials/prefill_authorship.jsonl",
    layer: int = 5,
    concept: str = "test",
    strength: float = 1.0,
    condition: str = "self_generated",
    seed: int = 42,
    max_new_tokens: int = 100
):
    setup_logging()
    logger = logging.getLogger(__name__)
    
    cfg = Config(
        model_name=model,
        concept_store_path=concept_store,
        out_path=out,
        layer=layer,
        concept=concept,
        strength=strength,
        condition=condition,
        seed=seed,
        max_new_tokens=max_new_tokens
    )
    
    logger.info(f"Starting trial with config: {cfg}")
    
    try:
        trial = run_trial(cfg)
        
        jsonl = JSONLLogger(cfg.out_path)
        jsonl.log(trial)
        jsonl.close()
        
        logger.info("Trial completed successfully")
    except Exception as e:
        logger.error(f"Trial failed: {e}", exc_info=True)

if __name__ == "__main__":
    fire.Fire(main)
