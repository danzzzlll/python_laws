from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from utils.base.GenerativeBase import GenerativeBaseModel
from config import settings

if settings.model_name == "qwen" and settings.model_name == "gemma":
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
elif settings.model_name == "llama":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


class QwenModel(GenerativeBaseModel):
    def __init__(self, model_name, model_directory, system_prompt=None, device_assignment='auto', computation_device='cuda'):
        super().__init__(model_name, system_prompt)

        self.model_directory = model_directory
        self.device_assignment = device_assignment
        self.computation_device = computation_device
        self.initialize_model()


    def initialize_model(self):
        """Initialize the model and tokenizer from the provided directory."""
        quantization_configuration = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_directory,
            device_map=self.device_assignment,
            quantization_config=quantization_configuration,
            trust_remote_code=True,
        ).to(self.computation_device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_directory,
        )


    def format_chat_template(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return full_prompt, self.tokenizer([full_prompt], return_tensors="pt").to(self.model.device)

    
    def generate_text(
            self, 
            system_prompt,
            user_prompt, 
            top_p=0.6, 
            temperature=0.8, 
            repetition_penalty=1.0, 
            max_tokens=1000, 
            skip_special_tokens=False, 
            sample=True,
        ):

        full_prompt, model_inputs = self.format_chat_template(system_prompt=system_prompt, user_prompt=user_prompt)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **model_inputs,
                max_length=max_tokens,
                do_sample=sample,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=skip_special_tokens)
        return output_text.strip()