#!/usr/bin/env python3
"""
Performance testing script for AsyncEngine with 392x512 token prompt.
This script launches an AsyncEngine with the same initialization pattern as 
CustomSGLangRollout and measures performance with a large prompt.
"""

import asyncio
import logging
import os
import time
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from sglang.srt.utils import get_ip, get_open_port

# Import the AsyncEngine from the custom rollout module
from verl.workers.rollout.sglang_rollout.sglang_rollout_custom import AsyncEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class PerformanceTester:
    def __init__(self, model_path: str = "Qwen/Qwen3-1.7B", **engine_kwargs):
        """
        Initialize the performance tester with AsyncEngine.
        
        Args:
            model_path: Path to the model to load
            **engine_kwargs: Additional engine configuration arguments
        """
        self.model_path = model_path
        self.engine_kwargs = engine_kwargs
        self.engine: Optional[AsyncEngine] = None
        
    async def initialize_engine(self):
        """Initialize the AsyncEngine with same pattern as CustomSGLangRollout."""
        logger.info("Initializing AsyncEngine...")
        
        # Default configuration matching CustomSGLangRollout initialization
        default_config = {
            "dtype": "auto",
            "mem_fraction_static": 0.9,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": 1,
            "node_rank": 0,
            "load_format": "auto",
            "dist_init_addr": None,
            "nnodes": 1,
            "trust_remote_code": True,
            "context_length": 8192,
            "port": 30000,
            "log_level": "INFO",
            "enable_metrics": True,
            "mm_attention_backend": "fa3",
            "attention_backend": "flashinfer",
            "skip_tokenizer_init": False,
            "max_running_requests": 10000,
            "schedule_policy": "fcfs",
            "schedule_conservativeness": 0.3,
        }
        
        # Update with any provided engine kwargs
        config = {**default_config, **self.engine_kwargs}
        
        logger.info(f"Engine configuration: {config}")
        
        # Initialize the AsyncEngine
        self.engine = AsyncEngine(
            model_path=self.model_path,
            **config
        )
        
        # Resume memory occupation (required for proper initialization)
        await self.engine.resume_memory_occupation()
        
        logger.info("AsyncEngine initialized successfully!")
    
    def generate_large_prompt(self, batch_size: int = 392, sequence_length: int = 512) -> list[list[int]]:
        """
        Generate a large prompt with specified dimensions.
        
        Args:
            batch_size: Number of sequences (392)
            sequence_length: Length of each sequence (512)
            
        Returns:
            List of token ID lists representing the prompt batch
        """
        logger.info(f"Generating prompt with shape ({batch_size}, {sequence_length})")
        
        # Generate random token IDs (avoiding special tokens like 0, 1, 2)
        # Using a range that's typically safe for most tokenizers
        min_token_id = 10
        max_token_id = 1000
        
        prompts = []
        for i in range(batch_size):
            # Generate random token sequence
            token_ids = torch.randint(
                min_token_id, 
                max_token_id, 
                (sequence_length,)
            ).tolist()
            prompts.append(token_ids)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{batch_size} prompt sequences")
        
        logger.info(f"Generated {batch_size} prompt sequences of length {sequence_length}")
        return prompts
    
    async def run_performance_test(
        self, 
        batch_size: int = 392, 
        sequence_length: int = 512,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1
    ):
        """
        Run the performance test with the specified parameters.
        
        Args:
            batch_size: Number of sequences in the batch
            sequence_length: Length of each input sequence
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Call initialize_engine() first.")
        
        logger.info("=" * 60)
        logger.info("STARTING PERFORMANCE TEST")
        logger.info("=" * 60)
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Sequence length: {sequence_length}")
        logger.info(f"Max new tokens: {max_new_tokens}")
        logger.info(f"Total input tokens: {batch_size * sequence_length:,}")
        
        # Generate the large prompt
        start_time = time.time()
        prompts = self.generate_large_prompt(batch_size, sequence_length)
        prompt_gen_time = time.time() - start_time
        logger.info(f"Prompt generation time: {prompt_gen_time:.2f} seconds")
        
        # Prepare sampling parameters
        sampling_params = {
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
            "min_new_tokens": max_new_tokens,
            "temperature": 1,
        }
        
        logger.info(f"Sampling parameters: {sampling_params}")
        
        # Run the generation
        logger.info("Starting generation...")
        generation_start_time = time.time()
        
        try:
            results = await self.engine.async_generate(
                prompt=None,  # Using input_ids instead
                input_ids=prompts,
                sampling_params=sampling_params,
                return_logprob=False,
            )
            
            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time
            
            # Calculate performance metrics
            total_input_tokens = batch_size * sequence_length
            total_output_tokens = len(results) * max_new_tokens if isinstance(results, list) else max_new_tokens
            total_tokens = total_input_tokens + total_output_tokens
            
            tokens_per_second = total_tokens / generation_time
            input_tokens_per_second = total_input_tokens / generation_time
            
            logger.info("=" * 60)
            logger.info("PERFORMANCE RESULTS")
            logger.info("=" * 60)
            logger.info(f"Generation time: {generation_time:.2f} seconds")
            logger.info(f"Total input tokens: {total_input_tokens:,}")
            logger.info(f"Total output tokens: {total_output_tokens:,}")
            logger.info(f"Total tokens processed: {total_tokens:,}")
            logger.info(f"Tokens per second: {tokens_per_second:.2f}")
            logger.info(f"Input tokens per second: {input_tokens_per_second:.2f}")
            logger.info(f"Batch processing rate: {batch_size / generation_time:.2f} sequences/second")
            
            # Log first few results for verification
            if isinstance(results, list) and len(results) > 0:
                logger.info(f"Successfully generated {len(results)} responses")
                logger.info("Sample result keys:", list(results[0].keys()) if results[0] else "No keys")
            else:
                logger.info("Single result generated")
                logger.info("Result keys:", list(results.keys()) if hasattr(results, 'keys') else "No keys")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
        finally:
            # Flush cache to clean up
            logger.info("Flushing engine cache...")
            await self.engine.flush_cache()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.engine:
            logger.info("Cleaning up engine...")
            await self.engine.flush_cache()
            # Additional cleanup if needed
            logger.info("Cleanup completed")


async def main():
    """Main function to run the performance test."""
    # Configuration - you can modify these parameters
    MODEL_PATH = "Qwen/Qwen3-1.7B"  # Change this to your desired model
    BATCH_SIZE = 392
    SEQUENCE_LENGTH = 3424
    MAX_NEW_TOKENS = 8192 - SEQUENCE_LENGTH - 1
    
    # Engine configuration (matching CustomSGLangRollout defaults)
    engine_config = {
        "context_length": 8192,
        "tp_size": 1,  # Adjust based on your GPU setup
        "mem_fraction_static": 0.9,
        "enable_metrics": True,
        "log_level": "INFO",
    }
    
    logger.info("Starting AsyncEngine Performance Test")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Target prompt shape: ({BATCH_SIZE}, {SEQUENCE_LENGTH})")
    
    # Initialize tester
    tester = PerformanceTester(model_path=MODEL_PATH, **engine_config)
    
    try:
        # Initialize engine
        await tester.initialize_engine()
        
        # Run performance test
        await tester.run_performance_test(
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.2,  # Matching config values
            top_p=1.0,
            top_k=-1
        )
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        await tester.cleanup()
        logger.info("Performance test completed")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
