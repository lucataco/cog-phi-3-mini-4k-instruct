# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
import subprocess
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import TextIteratorStreamer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_CACHE = "model-cache"
MODEL_URL = "https://weights.replicate.delivery/default/microsoft/Phi-3-mini-4k-instruct/model.tar"
MESSAGES="""<|system|>
{sys_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>
"""

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Download the model and tokenizer if they are not already cached
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="flash_attention_2",
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)

    def predict(
        self,
        prompt: str = Input(description="Text prompt to send to the model."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens.",
            ge=1,
            le=4096,
            default=200,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.",
            ge=0.1,
            le=5.0,
            default=0.1,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.",
            default=1,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=10.0,
            default=1.1,
        ),
        system_prompt: str = Input(
            description="System prompt.",
            default="You are a helpful AI assistant."
        ),
        seed: int = Input(
            description="The seed for the random number generator", default=None
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = torch.randint(0, 100000, (1,)).item()
        torch.random.manual_seed(seed)
        chat_format = MESSAGES.format(sys_prompt=system_prompt, user_prompt=prompt)
        formatted_prompt = self.tokenizer.decode(self.tokenizer(chat_format)["input_ids"])
        tokens = self.tokenizer(formatted_prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, remove_start_token=True)
        input_ids = tokens.input_ids.to(device=self.device)
        max_length = input_ids.shape[1] + max_length
        generation_kwargs = dict(
            input_ids=input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            do_sample=True
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for _, new_text in enumerate(streamer):
            yield new_text
        thread.join()