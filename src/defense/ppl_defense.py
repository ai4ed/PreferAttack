"""
Perplexity-based Defense (PPL-D) Implementation
Uses GPT-2 model to calculate perplexity and filter malicious responses
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from pathlib import Path

try:
    from utils.data_types import PairwiseExample, JudgeResponse
    from utils.judge import BaseJudge
except ImportError:
    from utils.data_types import PairwiseExample, JudgeResponse
    from utils.judge import BaseJudge


class PPLDefense:
    """Perplexity-based Defense using a language model for perplexity calculation"""
    
    def __init__(
        self,
        ppl_model_path: str = "/share/disk/llm_cache/gpt2",
        device: str = "cuda",
        threshold_multiplier: float = 2.0,
        threshold: Optional[float] = None,
            threshold_method: str = "robust",  # "mean_std", "robust", "percentile", "iqr", "fpr_based"
            target_fpr: float = 0.01  # Target False Positive Rate for "fpr_based" method (default 1%)
    ):
        """
        Initialize PPL Defense
        
        Args:
            ppl_model_path: Path to model for perplexity calculation (GPT-2, Gemma, etc.)
            device: Device to run the model on
            threshold_multiplier: Multiplier for threshold calculation (c in T = μ + c * σ)
            threshold: Direct threshold value (if provided, overrides threshold_multiplier)
            threshold_method: Method for computing threshold:
                - "mean_std": T = μ + c * σ (standard method, sensitive to outliers)
                - "robust": T = median + c * MAD (robust to outliers, recommended)
                - "percentile": Use percentile-based threshold (e.g., 95th percentile)
                - "iqr": T = Q3 + c * IQR (interquartile range method)
                - "fpr_based": Use FPR-based threshold selection (percentile-based, ensures FPR ≤ target_fpr)
            target_fpr: Target False Positive Rate for "fpr_based" method (default 0.01 = 1%)
        """
        self.ppl_model_path = ppl_model_path
        self.device = device
        self.threshold_multiplier = threshold_multiplier
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.target_fpr = target_fpr
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model for perplexity calculation (supports GPT-2, Gemma, etc.)"""
        print(f"Loading PPL model from {self.ppl_model_path}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.ppl_model_path,
                trust_remote_code=True
            )
            
            # Set pad_token if not exists
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # Add a pad token if none exists
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Resolve target device index
            device_index = 0
            if isinstance(self.device, str):
                if self.device.startswith("cuda") and ":" in self.device:
                    try:
                        device_index = int(self.device.split(":")[1])
                    except ValueError:
                        device_index = 0
            
            # Determine appropriate dtype based on model type
            # Gemma models typically use bfloat16 or float16, GPT-2 uses float32
            model_name_lower = self.ppl_model_path.lower()
            if "gemma" in model_name_lower:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
            
            # Load model using AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.ppl_model_path,
                torch_dtype=torch_dtype,
                device_map={"": device_index},
                trust_remote_code=True
            ).eval()
            
            # Resize token embeddings if pad_token was added
            if self.tokenizer.pad_token_id is not None:
                if self.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
                    self.model.resize_token_embeddings(len(self.tokenizer))
            
            model_name = "Gemma" if "gemma" in model_name_lower else "GPT-2" if "gpt2" in model_name_lower else "Model"
            print(f"✅ PPL model ({model_name}) loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading PPL model: {e}")
            raise
    
    def calculate_perplexity(self, text: str, verbose: bool = False) -> float:
        """
        Calculate perplexity of a text using the PPL model
        
        Args:
            text: Input text to calculate perplexity for
            verbose: Whether to print error messages
            
        Returns:
            Perplexity value (float)
        """
        if not text or not text.strip():
            return float('inf')  # Empty text has infinite perplexity
        
        try:
            # Tokenize the text with safer options
            # Use add_special_tokens=False to avoid potential issues with special tokens
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=False,
                add_special_tokens=True  # Keep special tokens but validate them
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            
            # Get actual vocab size from model config (most reliable)
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                # Fallback to tokenizer vocab size
                vocab_size = getattr(self.tokenizer, 'vocab_size', len(self.tokenizer))
                if vocab_size == 0:
                    vocab_size = 50257  # GPT-2 default vocab size
            
            # Validate and clean token IDs before processing
            # Filter out any invalid token IDs (out of vocabulary range)
            valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
            if not torch.all(valid_mask):
                # Replace invalid tokens with a safe token (e.g., padding or eos token)
                safe_token_id = getattr(self.tokenizer, 'pad_token_id', None) or getattr(self.tokenizer, 'eos_token_id', 0)
                if safe_token_id is None or safe_token_id >= vocab_size:
                    safe_token_id = 0  # Fallback to token 0
                input_ids = torch.where(valid_mask, input_ids, torch.tensor(safe_token_id, device=input_ids.device, dtype=input_ids.dtype))
                if verbose:
                    invalid_count = torch.sum(~valid_mask).item()
                    print(f"⚠️ Replaced {invalid_count} invalid token IDs with safe token {safe_token_id}")
            
            # Ensure input_ids is not empty
            if input_ids.numel() == 0 or input_ids.size(1) == 0:
                return float('inf')
            
            # Calculate log probabilities manually to avoid label indexing issues
            with torch.no_grad():
                try:
                    # Get model outputs without labels to avoid indexing issues
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Ensure shift_labels are within valid range
                    shift_labels = torch.clamp(shift_labels, min=0, max=vocab_size - 1)
                    
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    
                    # Calculate cross entropy loss manually
                    # Use log_softmax + nll_loss for numerical stability
                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    # Select log probabilities for the actual tokens
                    # Clamp labels again before gather to be extra safe
                    shift_labels = torch.clamp(shift_labels, min=0, max=log_probs.size(-1) - 1)
                    nll = -log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                    # Average over all tokens (ignore any NaN or inf values)
                    nll = nll[torch.isfinite(nll)]
                    if nll.numel() == 0:
                        return float('inf')
                    nll = nll.mean().item()
                    
                except (RuntimeError, IndexError, ValueError) as e:
                    error_str = str(e)
                    if "CUDA" in error_str or "cuda" in error_str.lower() or "index" in error_str.lower() or "out of bounds" in error_str.lower():
                        # Clear CUDA cache and try to recover
                        torch.cuda.empty_cache()
                        if verbose:
                            print(f"⚠️ CUDA/index error during perplexity calculation, clearing cache")
                        return float('inf')
                    raise
            
            # Perplexity = exp(nll)
            ppl = np.exp(nll)
            
            # Validate the result
            if not np.isfinite(ppl) or ppl <= 0:
                return float('inf')
            
            return ppl
            
        except Exception as e:
            if verbose:
                print(f"⚠️ Error calculating perplexity: {e}")
            # Clear CUDA cache on any error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float('inf')  # Return high perplexity on error
    
    def calculate_perplexity_batch(self, texts: List[str], verbose: bool = False) -> List[float]:
        """
        Calculate perplexity for a batch of texts
        
        Args:
            texts: List of texts to calculate perplexity for
            verbose: Whether to print error messages
            
        Returns:
            List of perplexity values
        """
        ppls = []
        error_count = 0
        for i, text in enumerate(texts):
            ppl = self.calculate_perplexity(text, verbose=(verbose and error_count < 5))
            ppls.append(ppl)
            if ppl == float('inf'):
                error_count += 1
        
        # Print summary if there were many errors
        if error_count > 0 and verbose:
            print(f"⚠️ {error_count}/{len(texts)} texts failed perplexity calculation")
        
        return ppls
    
    def compute_threshold_from_clean_data(
        self,
        clean_examples: List[PairwiseExample],
        num_samples: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """
        Compute threshold from clean data distribution
        
        Args:
            clean_examples: List of clean pairwise examples
            num_samples: Number of samples to use (None for all)
            
        Returns:
            (mean, std, threshold): Mean, standard deviation, and computed threshold
        """
        print("Computing PPL threshold from clean data...")
        
        # Limit number of samples if specified
        if num_samples is not None and num_samples < len(clean_examples):
            import random
            random.seed(42)
            clean_examples = random.sample(clean_examples, num_samples)
        
        # Collect all response texts
        all_texts = []
        for example in clean_examples:
            all_texts.append(example.response_a)
            all_texts.append(example.response_b)
        
        print(f"Calculating PPL for {len(all_texts)} clean responses...")
        
        # Calculate perplexities (suppress verbose errors during batch processing)
        ppls = self.calculate_perplexity_batch(all_texts, verbose=False)
        
        # Filter out infinite values and NaN values
        valid_ppls = [ppl for ppl in ppls if ppl != float('inf') and np.isfinite(ppl) and ppl > 0]
        
        if not valid_ppls:
            error_rate = (len(all_texts) - len(valid_ppls)) / len(all_texts) if all_texts else 1.0
            raise ValueError(
                f"No valid perplexity values computed from clean data "
                f"({len(valid_ppls)}/{len(all_texts)} succeeded, {error_rate*100:.1f}% failed). "
                f"This may be due to CUDA errors or invalid token sequences."
            )
        
        # Warn if too many failed
        if len(valid_ppls) < len(all_texts) * 0.5:
            print(f"⚠️ Warning: Only {len(valid_ppls)}/{len(all_texts)} ({len(valid_ppls)/len(all_texts)*100:.1f}%) "
                  f"perplexity calculations succeeded. Results may be less reliable.")
        
        # Calculate threshold based on selected method
        valid_ppls_array = np.array(valid_ppls)
        
        if self.threshold_method == "mean_std":
            # Standard method: T = μ + c * σ
            mean_ppl = np.mean(valid_ppls_array)
            std_ppl = np.std(valid_ppls_array)
            threshold = mean_ppl + self.threshold_multiplier * std_ppl
            print(f"✅ PPL statistics from clean data (mean_std method):")
            print(f"   Mean: {mean_ppl:.2f}")
            print(f"   Std: {std_ppl:.2f}")
            print(f"   Threshold (μ + {self.threshold_multiplier}σ): {threshold:.2f}")
            return mean_ppl, std_ppl, threshold
            
        elif self.threshold_method == "robust":
            # Robust method: T = median + c * MAD (Median Absolute Deviation)
            median_ppl = np.median(valid_ppls_array)
            mad = np.median(np.abs(valid_ppls_array - median_ppl))
            # Scale MAD to approximate standard deviation (for normal distribution, MAD ≈ 0.6745 * σ)
            # So we adjust the multiplier accordingly
            adjusted_multiplier = self.threshold_multiplier / 0.6745
            threshold = median_ppl + adjusted_multiplier * mad
            mean_ppl = np.mean(valid_ppls_array)
            std_ppl = np.std(valid_ppls_array)
            print(f"✅ PPL statistics from clean data (robust method):")
            print(f"   Mean: {mean_ppl:.2f}, Median: {median_ppl:.2f}")
            print(f"   Std: {std_ppl:.2f}, MAD: {mad:.2f}")
            print(f"   Threshold (median + {adjusted_multiplier:.2f} * MAD): {threshold:.2f}")
            return mean_ppl, std_ppl, threshold
            
        elif self.threshold_method == "percentile":
            # Percentile-based method: Use (100 - multiplier*5)th percentile
            # multiplier=2.0 -> 90th percentile, multiplier=3.0 -> 85th percentile
            percentile = max(50, min(99, 100 - self.threshold_multiplier * 5))
            threshold = np.percentile(valid_ppls_array, percentile)
            mean_ppl = np.mean(valid_ppls_array)
            std_ppl = np.std(valid_ppls_array)
            print(f"✅ PPL statistics from clean data (percentile method):")
            print(f"   Mean: {mean_ppl:.2f}")
            print(f"   Std: {std_ppl:.2f}")
            print(f"   Threshold ({percentile:.1f}th percentile): {threshold:.2f}")
            return mean_ppl, std_ppl, threshold
            
        elif self.threshold_method == "iqr":
            # IQR method: T = Q3 + c * IQR
            q1 = np.percentile(valid_ppls_array, 25)
            q3 = np.percentile(valid_ppls_array, 75)
            iqr = q3 - q1
            threshold = q3 + self.threshold_multiplier * iqr
            mean_ppl = np.mean(valid_ppls_array)
            std_ppl = np.std(valid_ppls_array)
            print(f"✅ PPL statistics from clean data (IQR method):")
            print(f"   Mean: {mean_ppl:.2f}")
            print(f"   Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
            print(f"   Threshold (Q3 + {self.threshold_multiplier} * IQR): {threshold:.2f}")
            return mean_ppl, std_ppl, threshold
            
        elif self.threshold_method == "fpr_based":
            # FPR-based method: Use log-perplexity and select threshold to ensure FPR ≤ target_fpr
            # Following the paper: calculate log-perplexity values and choose threshold
            # that ensures FPR ≤ target_fpr (default 1%)
            
            mean_ppl = np.mean(valid_ppls_array)
            std_ppl = np.std(valid_ppls_array)
            
            # FPR-based calculation (always use this method, no auto-fallback)
            # Calculate log-perplexity (log of perplexity)
            log_ppls = np.log(valid_ppls_array)
            
            # Calculate percentile for target FPR
            # FPR = 1% means 1% of clean samples will be above threshold
            # So threshold should be at (100 - target_fpr * 100)th percentile
            percentile = (1.0 - self.target_fpr) * 100  # For FPR=1%, this is 99th percentile
            
            # Get threshold in log-perplexity space
            log_threshold = np.percentile(log_ppls, percentile)
            
            # Convert back to perplexity threshold
            threshold = np.exp(log_threshold)
            
            # Calculate actual FPR (percentage of clean samples above threshold)
            actual_fpr = np.sum(valid_ppls_array > threshold) / len(valid_ppls_array)
            
            mean_log_ppl = np.mean(log_ppls)
            std_log_ppl = np.std(log_ppls)
            
            print(f"✅ PPL statistics from clean data (FPR-based method):")
            print(f"   Mean PPL: {mean_ppl:.2f}, Std PPL: {std_ppl:.2f}")
            print(f"   Mean log-PPL: {mean_log_ppl:.2f}, Std log-PPL: {std_log_ppl:.2f}")
            print(f"   Target FPR: {self.target_fpr*100:.2f}%")
            print(f"   Threshold (log-PPL {percentile:.1f}th percentile): {log_threshold:.2f}")
            print(f"   Threshold (PPL): {threshold:.2f}")
            print(f"   Actual FPR: {actual_fpr*100:.2f}% ({np.sum(valid_ppls_array > threshold)}/{len(valid_ppls_array)} samples)")
            
            return mean_ppl, std_ppl, threshold
            
        else:
            raise ValueError(f"Unknown threshold_method: {self.threshold_method}. "
                           f"Choose from: 'mean_std', 'robust', 'percentile', 'iqr', 'fpr_based'")
    
    def set_threshold(self, threshold: float):
        """Set the threshold directly"""
        self.threshold = threshold
        print(f"✅ PPL threshold set to: {threshold:.2f}")
    
    def is_malicious(self, text: str) -> Tuple[bool, float]:
        """
        Check if a text is malicious based on perplexity
        
        Args:
            text: Text to check
            
        Returns:
            (is_malicious, ppl_value): True if malicious (PPL > threshold), and the PPL value
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold_from_clean_data() or set_threshold() first.")
        
        ppl = self.calculate_perplexity(text)
        is_malicious = ppl > self.threshold
        
        return is_malicious, ppl
    
    def filter_response(self, response: str) -> Tuple[bool, float]:
        """
        Filter a response based on perplexity
        
        Args:
            response: Response text to filter
            
        Returns:
            (should_filter, ppl_value): True if should be filtered (malicious), and the PPL value
        """
        return self.is_malicious(response)
    
    def filter_example(self, example: PairwiseExample) -> Tuple[bool, Dict[str, float]]:
        """
        Filter a pairwise example based on perplexity of both responses
        
        Args:
            example: PairwiseExample to filter
            
        Returns:
            (should_filter, ppl_dict): True if should be filtered, and PPL values for both responses
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold_from_clean_data() or set_threshold() first.")
        
        ppl_a, ppl_b = self.calculate_perplexity(example.response_a), self.calculate_perplexity(example.response_b)
        
        # Filter if either response has high perplexity
        should_filter = (ppl_a > self.threshold) or (ppl_b > self.threshold)
        
        ppl_dict = {
            'response_a_ppl': ppl_a,
            'response_b_ppl': ppl_b,
            'max_ppl': max(ppl_a, ppl_b)
        }
        
        return should_filter, ppl_dict


class PPLDefendedJudge:
    """Wrapper around a Judge that applies PPL defense before judging"""
    
    def __init__(
        self,
        judge: BaseJudge,
        ppl_defense: PPLDefense,
        filter_mode: str = "instruction",
        original_preference: Optional[int] = None
    ):
        """
        Initialize PPL-defended Judge
        
        Args:
            judge: Base judge to wrap
            ppl_defense: PPL defense instance
            filter_mode: How to filter - "instruction" (filter modified instruction),
                        "response" (filter responses), or "both"
            original_preference: Original preference to return when filtered (maintains original preference)
        """
        self.judge = judge
        self.ppl_defense = ppl_defense
        self.filter_mode = filter_mode
        self.original_preference = original_preference
        self.filtered_count = 0
        self.total_count = 0
    
    def set_original_preference(self, preference: int):
        """Set the original preference to return when filtered"""
        self.original_preference = preference
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        """
        Judge pairwise with PPL defense
        
        If the modified instruction or responses have high perplexity, 
        return a fallback response (maintain original preference)
        """
        self.total_count += 1
        
        # Check if we should filter
        should_filter = False
        ppl_info = {}
        
        if self.filter_mode in ["instruction", "both"]:
            # Check modified instruction if provided
            text_to_check = modified_instruction if modified_instruction else example.instruction
            is_malicious, ppl = self.ppl_defense.is_malicious(text_to_check)
            if is_malicious:
                should_filter = True
                ppl_info['instruction_ppl'] = ppl
        
        if self.filter_mode in ["response", "both"] and not should_filter:
            # Check responses
            should_filter, ppl_dict = self.ppl_defense.filter_example(example)
            ppl_info.update(ppl_dict)
        
        if should_filter:
            self.filtered_count += 1
            # Return fallback response (maintain original preference)
            # If original_preference is set, use it; otherwise get it from the original judge
            if self.original_preference is not None:
                fallback_preference = self.original_preference
            else:
                # Fallback: get original preference from judge without filtering
                try:
                    original_response = self.judge.judge_pairwise(example)
                    fallback_preference = original_response.preference
                except:
                    # Last resort: use the first response (preference 0)
                    fallback_preference = 0
            
            # Safely extract max PPL value from ppl_info
            ppl_values = [v for v in ppl_info.values() if isinstance(v, (int, float)) and not (isinstance(v, float) and (v == float('inf') or v != v))]
            if ppl_values:
                max_ppl = max(ppl_values)
                ppl_str = f"{max_ppl:.2f}"
            else:
                ppl_str = "N/A"
            # Verify that we're returning the original preference
            if self.original_preference is not None and fallback_preference != self.original_preference:
                import warnings
                warnings.warn(f"PPL Defense: Expected to return original preference {self.original_preference}, "
                            f"but returning {fallback_preference}. This may indicate a bug.")
            
            return JudgeResponse(
                preference=fallback_preference,  # Maintain original preference (defense should not change preference)
                confidence=0.1,  # Low confidence to indicate filtering
                raw_response=f"Filtered by PPL defense (PPL: {ppl_str})"
            )
        
        # If not filtered, proceed with normal judging
        return self.judge.judge_pairwise(example, modified_instruction)
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        filter_rate = self.filtered_count / self.total_count if self.total_count > 0 else 0.0
        return {
            'total_judgments': self.total_count,
            'filtered_count': self.filtered_count,
            'filter_rate': filter_rate
        }
    
    def reset_stats(self):
        """Reset filtering statistics"""
        self.filtered_count = 0
        self.total_count = 0

